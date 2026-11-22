"""One thin function every AI step uses to call the model through the OpenAI Agents SDK.

Provider configuration lives in ``pmp_config``. This module builds the provider client LAZILY, from that
configuration only, and injects it into the model object. Nothing here runs at import time: no
environment variable is read into a global, ``os.environ`` is never modified and the SDK's global default
client / tracing switches are never touched.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional, Type, TypeVar

from pydantic import BaseModel

from pmp_config import ConfigError, ProviderConfig, build_client, build_model, load_config, redact
from project_agents.json_output import LLMOutputError, parse_json_output  # noqa: F401  (re-exported)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")
LLMCall = Callable[[str, str, str, Type[T]], Awaitable[T]]
ProviderErrorKind = Literal["rate_limit", "timeout", "connection", "auth", "api_error"]

log = logging.getLogger("pmp.llm")
MAX_TOKEN_CAP = 8000          # a truncation retry never asks for more than this
MAX_RETRY_WAIT_SECONDS = 30.0
DEFAULT_RETRY_BACKOFF = 0.75
_RETRY_AFTER_RE = re.compile(r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|s)\b", re.I)
_LIMIT_HINT_RE = re.compile(r"\b(OTPM|ITPM|TPM)\)?\s*:\s*Limit\s+(\d+)", re.I)
_WINDOW_SECONDS = 60.0
_MAX_PACE_WAIT_SECONDS = 61.0


@dataclass(frozen=True)
class StageMetric:
    """Safe telemetry for one provider stage (contains zero credentials or user content)."""

    stage: str
    requests: int
    ms: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    mode: str


class TokenQuotaPacer:
    """60-second sliding-window request pacer tracking input (ITPM) and output (OTPM) tokens separately."""

    def __init__(
        self,
        itpm_limit: int = 0,
        otpm_limit: int = 0,
        *,
        clock_fn: Callable[[], float] = time.monotonic,
    ):
        self.itpm_limit = max(0, int(itpm_limit))
        self.otpm_limit = max(0, int(otpm_limit))
        self._clock = clock_fn
        self._events: deque[tuple[float, int, int]] = deque()
        self._lock = threading.Lock()

    def _prune_locked(self, now: float) -> None:
        cutoff = now - _WINDOW_SECONDS
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()

    def current_usage(self) -> tuple[int, int]:
        with self._lock:
            self._prune_locked(self._clock())
            return sum(e[1] for e in self._events), sum(e[2] for e in self._events)

    def adjust_limit(self, kind: str, observed_limit: int) -> None:
        if observed_limit <= 0:
            return
        safe_cap = max(100, int(observed_limit * 0.9))
        with self._lock:
            if kind == "otpm":
                self.otpm_limit = min(self.otpm_limit, safe_cap) if self.otpm_limit > 0 else safe_cap
            elif kind in ("itpm", "tpm"):
                self.itpm_limit = min(self.itpm_limit, safe_cap) if self.itpm_limit > 0 else safe_cap

    def record(self, input_tokens: int, output_tokens: int) -> None:
        in_tok = max(0, int(input_tokens))
        out_tok = max(0, int(output_tokens))
        if in_tok == 0 and out_tok == 0:
            return
        with self._lock:
            now = self._clock()
            self._prune_locked(now)
            self._events.append((now, in_tok, out_tok))

    def wait_needed(self, est_in: int, est_out: int) -> float:
        """Compute wait time needed so adding estimated input/output tokens fits within sliding budgets."""
        if self.itpm_limit <= 0 and self.otpm_limit <= 0:
            return 0.0
        with self._lock:
            now = self._clock()
            self._prune_locked(now)
            if not self._events:
                return 0.0
            used_in = sum(e[1] for e in self._events)
            used_out = sum(e[2] for e in self._events)
            need_in = self.itpm_limit > 0 and est_in < self.itpm_limit and (used_in + est_in) > self.itpm_limit
            need_out = self.otpm_limit > 0 and est_out < self.otpm_limit and (used_out + est_out) > self.otpm_limit
            if not (need_in or need_out):
                return 0.0

            # Find earliest timestamp whose eviction brings both counters back under budget.
            rem_in, rem_out = used_in, used_out
            release_at = self._events[0][0]
            for ts, ev_in, ev_out in self._events:
                rem_in -= ev_in
                rem_out -= ev_out
                release_at = ts
                ok_in = self.itpm_limit <= 0 or (rem_in + est_in) <= self.itpm_limit
                ok_out = self.otpm_limit <= 0 or (rem_out + est_out) <= self.otpm_limit
                if ok_in and ok_out:
                    break
            return max(0.0, min((release_at + _WINDOW_SECONDS) - now + 0.1, _MAX_PACE_WAIT_SECONDS))

    async def pace(self, est_in: int, est_out: int, sleep_fn: Callable[[float], Awaitable[Any]]) -> float:
        wait = self.wait_needed(est_in, est_out)
        if wait > 0:
            await sleep_fn(wait)
        return wait


class GuardrailTripped(Exception):
    """An SDK guardrail stopped an agent. kind is 'input' or 'output'."""

    def __init__(self, kind: str, info: Any):
        super().__init__(f"{kind} guardrail tripped: {info}")
        self.kind = kind
        self.info = info


class ProviderCallError(RuntimeError):
    """Sanitized failure when the live LLM provider could not fulfill a request."""

    def __init__(self, kind: ProviderErrorKind, user_message: str, *, retry_after: Optional[float] = None):
        super().__init__(user_message)
        self.kind = kind
        self.user_message = user_message
        self.retry_after = retry_after


# --------------------------------------------------------------------------- runtime (lazy, explicit)
@dataclass
class Runtime:
    """The live provider: configuration + injected client + SDK model. Built on demand."""

    config: ProviderConfig
    client: Any = None
    model: Any = None
    structured_mode: str = "prompt"
    worker_structured_mode: Optional[str] = None
    sleep_fn: Callable[[float], Awaitable[Any]] = asyncio.sleep
    pacer: TokenQuotaPacer = field(default_factory=TokenQuotaPacer)
    events: deque = field(default_factory=lambda: deque(maxlen=50))   # sanitized, observable diagnostics
    metrics: deque[StageMetric] = field(default_factory=lambda: deque(maxlen=100))

    def note(self, message: str) -> None:
        safe = redact(message, self.config.api_key)
        self.events.append(safe)
        log.warning(safe)

    def metrics_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "stage": m.stage,
                "requests": m.requests,
                "ms": m.ms,
                "in": m.input_tokens,
                "out": m.output_tokens,
                "mode": m.mode,
            }
            for m in self.metrics
        ]


_RUNTIME: Optional[Runtime] = None
_RUNTIME_LOCK = threading.Lock()


def configure_runtime(
    config: Optional[ProviderConfig] = None,
    *,
    client: Any = None,
    sleep_fn: Optional[Callable[[float], Awaitable[Any]]] = None,
    clock_fn: Optional[Callable[[], float]] = None,
) -> Runtime:
    """Explicitly (re)configure the provider. ``client`` lets callers/tests inject a prepared client."""
    cfg = config or load_config()
    if cfg.live and client is None:
        client = build_client(cfg)
    pacer = TokenQuotaPacer(
        itpm_limit=cfg.itpm_limit,
        otpm_limit=cfg.otpm_limit,
        clock_fn=clock_fn or time.monotonic,
    )
    rt = Runtime(
        config=cfg,
        client=client,
        model=build_model(cfg, client) if cfg.live else None,
        structured_mode=cfg.structured_output,
        sleep_fn=sleep_fn or asyncio.sleep,
        pacer=pacer,
    )
    global _RUNTIME
    with _RUNTIME_LOCK:
        _RUNTIME = rt
    return rt


def get_runtime() -> Runtime:
    """The current runtime; reads the environment (without modifying it) the first time it is needed."""
    with _RUNTIME_LOCK:
        rt = _RUNTIME
    return rt if rt is not None else configure_runtime()


def reset_runtime() -> None:
    global _RUNTIME
    with _RUNTIME_LOCK:
        _RUNTIME = None


# --------------------------------------------------------------------------- per-run settings
@dataclass
class RunSettings:
    model: Any = None        # an injected SDK model (tests use a scripted one); wins over the provider
    session: Any = None      # accepted for backward compatibility, NOT used: extraction is stateless
    context: Any = None


_SETTINGS: ContextVar[Optional[RunSettings]] = ContextVar("pmp_run_settings", default=None)


@contextmanager
def run_settings(**kw):
    token = _SETTINGS.set(RunSettings(**kw))
    try:
        yield
    finally:
        _SETTINGS.reset(token)


def get_settings() -> RunSettings:
    return _SETTINGS.get() or RunSettings()


async def get_active_model() -> Any:
    """The injected test model, or the configured provider's model. Never silently switches provider."""
    s = get_settings()
    if s.model is not None:
        return s.model
    rt = get_runtime()
    if not rt.config.live:
        raise ConfigError("No LLM provider is configured (offline mode). Set OPENAI_API_KEY or GROQ_API_KEY.")
    return rt.model


def make_run_config(cfg: Optional[ProviderConfig]):
    """Tracing uploads project data to OpenAI's dashboard: only when explicitly enabled for OpenAI."""
    from agents import RunConfig

    return RunConfig(tracing_disabled=not (cfg is not None and cfg.tracing))


def make_model_settings(
    cfg: ProviderConfig,
    max_tokens: int,
    mode: str,
    *,
    is_tool_agent: bool = False,
):
    """Token budget / structured-output / reasoning settings built from configuration."""
    from agents import ModelSettings

    extra: dict[str, Any] = {}
    kwargs: dict[str, Any] = {}
    if cfg.token_param == "max_completion_tokens":
        extra["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    if mode == "json_object":
        extra["response_format"] = {"type": "json_object"}
    if cfg.reasoning_effort:
        extra["reasoning_effort"] = cfg.reasoning_effort
    if is_tool_agent and cfg.provider == "groq":
        # Avoid parallel or high-temperature tool syntax failures on Groq models.
        kwargs["temperature"] = 0.0
        kwargs["parallel_tool_calls"] = False
    return ModelSettings(extra_args=extra or None, **kwargs)


def estimate_tokens(*texts: str) -> int:
    return max(1, sum(len(t or "") for t in texts) // 4)


def record_stage_usage(rt: Runtime, stage: str, result: Any, elapsed_ms: int, mode: str) -> None:
    """Record per-stage latency and token usage from provider metadata into runtime metrics & quota pacer."""
    wrapper = getattr(result, "context_wrapper", None)
    usage = getattr(wrapper, "usage", None)
    requests = int(getattr(usage, "requests", 0) or 0)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    total_tok = int(getattr(usage, "total_tokens", 0) or (in_tok + out_tok))
    rt.metrics.append(
        StageMetric(
            stage=stage,
            requests=requests or 1,
            ms=max(0, int(elapsed_ms)),
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=total_tok,
            mode=mode,
        )
    )
    rt.pacer.record(in_tok, out_tok)


def structured_rejected(exc: Exception) -> bool:
    """True when the provider refused the structured-output request itself (HTTP 400 about JSON/format)."""
    try:
        from openai import BadRequestError
    except Exception:  # pragma: no cover
        return False
    if not isinstance(exc, BadRequestError):
        return False
    text = str(exc).lower()
    return any(k in text for k in ("response_format", "json", "schema", "structured", "tool"))


def worker_mode_for(rt: Runtime) -> str:
    """Choose compatible structured output mode for tool-calling workers without stripping non-tool JSON mode."""
    if rt.worker_structured_mode is not None:
        return rt.worker_structured_mode
    if rt.config.provider == "groq" and rt.structured_mode == "json_object":
        # Groq rejects response_format={"type": "json_object"} when tools are attached.
        rt.worker_structured_mode = "prompt"
        rt.note("Worker tools: using prompt-validated JSON for groq tool-calling compatibility "
                "(non-tool calls keep json_object)")
        return "prompt"
    return rt.structured_mode


def _extract_retry_after(exc: Exception) -> Optional[float]:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        raw_ms = headers.get("retry-after-ms")
        if raw_ms is not None:
            try:
                val = float(raw_ms) / 1000.0
                if val >= 0:
                    return val
            except (TypeError, ValueError):
                pass
        raw = headers.get("retry-after")
        if raw is not None:
            try:
                val = float(raw)
                if val >= 0:
                    return val
            except (TypeError, ValueError):
                pass
    m = _RETRY_AFTER_RE.search(str(exc))
    if m:
        val = float(m.group(1))
        if m.group(2).lower() == "ms":
            val /= 1000.0
        if val >= 0:
            return val
    return None


def _classify_provider_error(exc: Exception, provider: str) -> Optional[tuple[ProviderErrorKind, bool, str, Optional[float]]]:
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, RateLimitError
    except Exception:  # pragma: no cover
        return None

    retry_after = _extract_retry_after(exc)
    if isinstance(exc, AuthenticationError):
        return (
            "auth",
            False,
            f"Authentication failed with {provider}. Check your API key configuration. Nothing was executed.",
            None,
        )
    if isinstance(exc, RateLimitError):
        wait_hint = f" Wait ~{math.ceil(retry_after)}s and try again." if retry_after else " Please wait a moment and try again."
        return (
            "rate_limit",
            True,
            f"Live provider rate limit reached ({provider}).{wait_hint} Nothing was executed.",
            retry_after,
        )
    if isinstance(exc, APITimeoutError):
        return (
            "timeout",
            True,
            f"Live provider request timed out ({provider}). Please try again. Nothing was executed.",
            None,
        )
    if isinstance(exc, APIConnectionError):
        return (
            "connection",
            True,
            f"Could not connect to live provider ({provider}). Check network access and try again. Nothing was executed.",
            None,
        )
    if isinstance(exc, APIStatusError):
        retryable = getattr(exc, "status_code", 0) >= 500
        return (
            "api_error",
            retryable,
            f"Live provider returned an error ({provider}, HTTP {getattr(exc, 'status_code', '?')}). Nothing was executed.",
            retry_after if retryable else None,
        )
    return None


async def run_provider_call(rt: Runtime, name: str, operation: Callable[[], Awaitable[R]]) -> R:
    """Execute one provider call under a single bounded transient retry budget (respecting Retry-After)."""
    max_retries = max(0, int(rt.config.provider_retries))
    attempt = 0
    while True:
        attempt += 1
        try:
            return await operation()
        except Exception as exc:
            if structured_rejected(exc):
                raise
            info = _classify_provider_error(exc, rt.config.provider)
            if info is None:
                raise
            kind, retryable, user_msg, retry_after = info
            if kind == "rate_limit":
                m_lim = _LIMIT_HINT_RE.search(str(exc))
                if m_lim:
                    rt.pacer.adjust_limit(m_lim.group(1).lower(), int(m_lim.group(2)))
            if retryable and attempt <= max_retries:
                if retry_after is not None and retry_after > MAX_RETRY_WAIT_SECONDS:
                    rt.note(f"{name}: provider asked to wait {retry_after:.1f}s (>{MAX_RETRY_WAIT_SECONDS:.0f}s cap); aborting retries")
                    raise ProviderCallError(kind, user_msg, retry_after=retry_after) from exc
                delay = (
                    min(retry_after + 0.25, MAX_RETRY_WAIT_SECONDS)
                    if retry_after is not None
                    else min(DEFAULT_RETRY_BACKOFF * (2 ** (attempt - 1)), MAX_RETRY_WAIT_SECONDS)
                )
                rt.note(f"{name}: transient {kind} ({type(exc).__name__}), retrying {attempt}/{max_retries} in {delay:.2f}s")
                await rt.sleep_fn(delay)
                continue
            rt.note(f"{name}: provider call failed ({type(exc).__name__}): {exc}")
            raise ProviderCallError(kind, user_msg, retry_after=retry_after) from exc


def _retry_note(err: LLMOutputError) -> str:
    return (f"\n\nYour previous reply was rejected ({err.kind}: {redact(err)}). "
            "Reply again with ONE complete, valid JSON object only - no prose, no markdown.")


# --------------------------------------------------------------------------- the call
async def call_llm(name: str, instructions: str, prompt: str, output_type: Type[T]) -> T:
    from agents import Agent, Runner
    from agents.exceptions import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered

    from project_agents.sdk_guardrails import guardrails_for

    s = get_settings()
    in_guards, out_guards = guardrails_for(name)
    kwargs: dict = {}
    if s.context is not None:
        kwargs["context"] = s.context

    # Injected model (pytest scripted model / custom model): the SDK's native typed output path.
    if s.model is not None:
        agent = Agent(name=name, instructions=instructions, model=s.model, output_type=output_type,
                      input_guardrails=in_guards, output_guardrails=out_guards)
        try:
            result = await Runner.run(agent, prompt, run_config=make_run_config(None), **kwargs)
            out = result.final_output
            return output_type.model_validate(out) if isinstance(out, dict) else out
        except InputGuardrailTripwireTriggered as exc:
            raise GuardrailTripped("input", exc.guardrail_result.output.output_info) from exc
        except OutputGuardrailTripwireTriggered as exc:
            raise GuardrailTripped("output", exc.guardrail_result.output.output_info) from exc

    rt = get_runtime()
    cfg = rt.config
    if not cfg.live:
        raise ConfigError("No LLM provider is configured (offline mode). Set OPENAI_API_KEY or GROQ_API_KEY.")

    max_tokens, attempt, downgraded, last_err = cfg.max_output_tokens, 0, False, None
    current_prompt = prompt
    while attempt < cfg.max_attempts:
        attempt += 1
        mode = rt.structured_mode

        async def _attempt(p: str = current_prompt, m: str = mode, mt: int = max_tokens) -> T:
            return await _call_once(
                rt, name, instructions, p, output_type, m, mt, in_guards, kwargs
            )

        try:
            out = await run_provider_call(rt, name, _attempt)
            break
        except InputGuardrailTripwireTriggered as exc:
            raise GuardrailTripped("input", exc.guardrail_result.output.output_info) from exc
        except ProviderCallError:
            raise
        except LLMOutputError as err:
            last_err = err
            if err.kind == "truncated":
                max_tokens = min(max_tokens * 2, MAX_TOKEN_CAP)
            rt.note(f"{name}: {err.kind} model output on attempt {attempt}/{cfg.max_attempts}: {redact(err, cfg.api_key)}")
            current_prompt = prompt + _retry_note(err)
        except Exception as exc:
            if mode != "prompt" and not downgraded and structured_rejected(exc):
                downgraded = True
                rt.structured_mode = "prompt"
                rt.note(f"{name}: provider rejected structured output mode '{mode}' for model '{cfg.model}'; "
                        "falling back to prompt-only JSON (explicit, per-runtime)")
                attempt -= 1            # the downgrade itself does not consume a retry
                continue
            rt.note(f"{name}: provider call failed ({type(exc).__name__}): {exc}")
            raise
    else:
        assert last_err is not None
        raise LLMOutputError(last_err.kind, f"{name}: no usable output after {cfg.max_attempts} attempt(s): "
                                            f"{redact(last_err, cfg.api_key)}")

    for g in out_guards:
        fn = getattr(g, "guardrail_function", None)
        if fn:
            res_g = fn(s.context, None, out)
            if hasattr(res_g, "__await__"):
                res_g = await res_g
            if getattr(res_g, "tripwire_triggered", False):
                raise GuardrailTripped("output", getattr(res_g, "output_info", None))
    return out


_STAGE_OUT_ESTIMATES = {
    "Project Orchestrator": 120,
    "Evaluator": 200,
    "Extraction Agent": 280,
}


def _estimate_stage_output(name: str, max_tokens: int, otpm_limit: int) -> int:
    if otpm_limit <= 0:
        return min(200, max_tokens)
    base = _STAGE_OUT_ESTIMATES.get(name, 500)
    return min(base, max_tokens, max(1, otpm_limit - 1))


async def _call_once(rt: Runtime, name, instructions, prompt, output_type, mode, max_tokens, in_guards, kwargs):
    from agents import Agent, AgentOutputSchema, Runner
    from agents.exceptions import ModelBehaviorError

    cfg = rt.config
    settings = make_model_settings(cfg, max_tokens, mode)
    est_in = estimate_tokens(instructions, prompt)
    est_out = _estimate_stage_output(name, max_tokens, cfg.otpm_limit)
    waited = await rt.pacer.pace(est_in, est_out, rt.sleep_fn)
    if waited > 0:
        rt.note(f"{name}: paced {waited:.2f}s to respect input/output token limits")

    t0 = time.perf_counter()
    if mode == "json_schema":
        agent = Agent(name=name, instructions=instructions, model=rt.model,
                      output_type=AgentOutputSchema(output_type, strict_json_schema=True),
                      input_guardrails=in_guards, model_settings=settings)
        try:
            result = await Runner.run(agent, prompt, run_config=make_run_config(cfg), **kwargs)
        except ModelBehaviorError as exc:
            raise LLMOutputError("malformed", redact(exc, cfg.api_key)[:200]) from None
        record_stage_usage(rt, name, result, round((time.perf_counter() - t0) * 1000), mode)
        out = result.final_output
        return output_type.model_validate(out) if isinstance(out, dict) else out

    schema_str = json.dumps(output_type.model_json_schema(), ensure_ascii=False)
    full_instructions = (
        f"{instructions}\n\nIMPORTANT: Return ONLY a valid JSON object matching this JSON schema "
        f"(no markdown fences, no extra text):\n{schema_str}"
    )
    agent = Agent(
        name=name,
        instructions=full_instructions,
        model=rt.model,
        input_guardrails=in_guards,
        model_settings=settings,
    )
    result = await Runner.run(agent, prompt, run_config=make_run_config(cfg), **kwargs)
    record_stage_usage(rt, name, result, round((time.perf_counter() - t0) * 1000), mode)
    return parse_json_output(result.final_output, output_type)
