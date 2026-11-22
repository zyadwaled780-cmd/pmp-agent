"""Central, side-effect-free LLM provider configuration.

Importing this module (or any module that uses it) never reads credentials into globals, never
modifies ``os.environ`` and never touches the OpenAI Agents SDK's global client settings.
Configuration is *read* from the environment on demand by ``load_config`` and *injected* at runtime
(``project_agents.llm`` builds an ``AsyncOpenAI`` client from it and hands it to the model object).

Provider selection (deterministic, documented in README.md):

1. ``PMP_PROVIDER=openai|groq|offline`` forces a provider. A forced provider whose credential is
   missing is a configuration ERROR (never a silent switch to another provider).
2. ``PMP_PROVIDER=auto`` (default, also when unset):
   * ``OPENAI_API_KEY`` set                  -> openai   (also when GROQ_API_KEY is set: OpenAI wins)
   * only ``GROQ_API_KEY`` set               -> groq
   * neither                                 -> offline  (rule-based demo mode, no network)

There is NO automatic cross-provider fallback. Credentials are only ever sent to the provider they
belong to: ``OPENAI_API_KEY`` -> OpenAI (or ``OPENAI_BASE_URL`` if you set it), ``GROQ_API_KEY`` -> Groq.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union
from urllib.parse import urlparse

PROVIDERS = ("openai", "groq", "offline")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "groq": "qwen/qwen3.8-27b"}
STRUCTURED_MODES = ("json_schema", "json_object", "prompt")
# OpenAI supports strict json_schema; Groq's support is model dependent, json_object is the portable mode.
DEFAULT_STRUCTURED = {"openai": "json_schema", "groq": "json_object"}
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")
TOKEN_PARAMS = ("max_tokens", "max_completion_tokens")
DEFAULT_SESSION_DB_PATH = Path(__file__).resolve().parent / ".pmp_sessions" / "sessions.db"
DEFAULT_DOTENV_PATHS = (Path(__file__).resolve().parent / ".env",)
_PLACEHOLDER_KEYS = frozenset({"your-openai-api-key-here", "your-groq-api-key-here"})

_SECRET_RE = re.compile(r"(gsk_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]{12,})")


class ConfigError(ValueError):
    """Invalid or incomplete configuration. The message never contains a credential."""


def redact(text: Any, *secrets: Optional[str]) -> str:
    """Mask credentials in any text before it is logged, shown or stored."""
    out = str(text)
    for s in secrets:
        if s and len(s) >= 6:
            out = out.replace(s, "***")
    return _SECRET_RE.sub("***", out)


@dataclass(frozen=True)
class PersistenceConfig:
    enabled: bool = False
    db_path: Path = DEFAULT_SESSION_DB_PATH


@dataclass(frozen=True)
class ProviderConfig:
    provider: str                       # openai | groq | offline
    model: Optional[str] = None
    api_key: Optional[str] = field(default=None, repr=False)   # never shown in repr / diagnostics
    base_url: Optional[str] = None
    structured_output: str = "prompt"   # json_schema | json_object | prompt
    timeout: float = 30.0
    max_output_tokens: int = 1500
    worker_max_tokens: int = 600
    max_attempts: int = 2               # bounded retries for malformed / truncated output
    provider_retries: int = 2           # unified retry budget for transient provider errors
    itpm_limit: int = 0                 # sliding-window input tokens/min budget (0 = disabled)
    otpm_limit: int = 0                 # sliding-window output tokens/min budget (0 = disabled)
    reasoning_effort: Optional[str] = None
    token_param: str = "max_tokens"
    tracing: bool = False
    selection_reason: str = ""

    @property
    def live(self) -> bool:
        return self.provider != "offline"

    def diagnostics(self) -> dict:
        """Safe to show / log: never contains the key, only whether one is configured."""
        host = urlparse(self.base_url).netloc if self.base_url else ("api.openai.com" if self.provider == "openai" else "")
        return {
            "provider": self.provider, "model": self.model, "endpoint_host": host,
            "api_key_configured": bool(self.api_key), "structured_output": self.structured_output,
            "max_output_tokens": self.max_output_tokens, "worker_max_tokens": self.worker_max_tokens,
            "max_attempts": self.max_attempts, "provider_retries": self.provider_retries,
            "itpm_limit": self.itpm_limit, "otpm_limit": self.otpm_limit,
            "reasoning_effort": self.reasoning_effort, "selection_reason": self.selection_reason,
        }

    def describe(self) -> str:
        d = self.diagnostics()
        if not self.live:
            return f"offline demo mode ({self.selection_reason})"
        return f"{d['provider']} / {d['model']} / structured={d['structured_output']} ({self.selection_reason})"


def _clean(env: Mapping[str, str], name: str) -> str:
    return (env.get(name) or "").strip()


def _clean_key(env: Mapping[str, str], name: str) -> str:
    val = _clean(env, name)
    return "" if val in _PLACEHOLDER_KEYS else val


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read a .env file purely into a dict without mutating os.environ."""
    if not path.is_file():
        return {}
    try:
        from dotenv import dotenv_values
        raw = dotenv_values(path)
        return {str(k): str(v) for k, v in raw.items() if k and v is not None}
    except ImportError:
        out: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            k, v = stripped.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                out[k] = v
        return out


def resolve_dotenv_path(dotenv_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Return the first existing .env file (explicit path, package root, project_agents/, or repo parent)."""
    if dotenv_path is not None:
        candidate = Path(dotenv_path)
        return candidate if candidate.is_file() else None
    for candidate in DEFAULT_DOTENV_PATHS:
        if candidate.is_file():
            return candidate
    return None


def _effective_env(
    env: Optional[Mapping[str, str]],
    dotenv_path: Optional[Union[str, Path]] = None,
) -> Mapping[str, str]:
    """Resolve effective config mapping without modifying os.environ.

    Explicit ``env`` overrides ``os.environ`` and bypasses implicit ``.env`` unless ``dotenv_path``
    is explicitly provided. Process environment variables always take precedence over ``.env``.
    """
    if env is not None and dotenv_path is None:
        return env
    base_env: Mapping[str, str] = os.environ if env is None else env
    path = resolve_dotenv_path(dotenv_path)
    dotenv_vals = _read_dotenv(path) if path is not None else {}
    if not dotenv_vals:
        return base_env
    merged = dict(dotenv_vals)
    merged.update({k: v for k, v in base_env.items() if v is not None})
    return merged


def _bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = _clean(env, name).lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be a boolean (true/false)")


def _int(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    raw = _clean(env, name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer between {lo} and {hi}") from None
    if not lo <= value <= hi:
        raise ConfigError(f"{name} must be an integer between {lo} and {hi}")
    return value


def _url(env: Mapping[str, str], name: str) -> Optional[str]:
    raw = _clean(env, name)
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ConfigError(f"{name} must be an http(s) URL")
    return raw.rstrip("/")


def load_persistence_config(
    env: Optional[Mapping[str, str]] = None,
    dotenv_path: Optional[Union[str, Path]] = None,
) -> PersistenceConfig:
    """Read optional durable session persistence settings (disabled by default)."""
    resolved = _effective_env(env, dotenv_path)
    enabled = _bool(resolved, "PMP_PERSIST_SESSIONS", default=False)
    raw_path = _clean(resolved, "PMP_SESSION_DB_PATH")
    db_path = Path(raw_path).expanduser() if raw_path else DEFAULT_SESSION_DB_PATH
    if enabled and not str(db_path).strip():
        raise ConfigError("PMP_SESSION_DB_PATH must not be empty when persistence is enabled")
    return PersistenceConfig(enabled=enabled, db_path=db_path)


def load_config(
    env: Optional[Mapping[str, str]] = None,
    dotenv_path: Optional[Union[str, Path]] = None,
) -> ProviderConfig:
    """Read configuration from ``env`` (default ``os.environ`` + local ``.env`` if present). Pure: nothing is mutated."""
    env = _effective_env(env, dotenv_path)
    requested = (_clean(env, "PMP_PROVIDER") or "auto").lower()
    if requested not in ("auto", *PROVIDERS):
        raise ConfigError("PMP_PROVIDER must be one of: auto, openai, groq, offline")

    openai_key, groq_key = _clean_key(env, "OPENAI_API_KEY"), _clean_key(env, "GROQ_API_KEY")
    if requested == "auto":
        if openai_key:
            provider = "openai"
            reason = ("auto: OPENAI_API_KEY and GROQ_API_KEY both set, OpenAI wins (set PMP_PROVIDER=groq to override)"
                      if groq_key else "auto: OPENAI_API_KEY is set")
        elif groq_key:
            provider, reason = "groq", "auto: only GROQ_API_KEY is set"
        else:
            provider, reason = "offline", "auto: no provider credentials found"
    else:
        provider, reason = requested, f"PMP_PROVIDER={requested}"
        if provider == "openai" and not openai_key:
            raise ConfigError("PMP_PROVIDER=openai but OPENAI_API_KEY is not set")
        if provider == "groq" and not groq_key:
            raise ConfigError("PMP_PROVIDER=groq but GROQ_API_KEY is not set")

    if provider == "offline":
        return ProviderConfig(provider="offline", selection_reason=reason)

    structured = _clean(env, "PMP_STRUCTURED_OUTPUT").lower() or DEFAULT_STRUCTURED[provider]
    if structured not in STRUCTURED_MODES:
        raise ConfigError("PMP_STRUCTURED_OUTPUT must be one of: " + ", ".join(STRUCTURED_MODES))
    effort = _clean(env, "PMP_REASONING_EFFORT").lower() or None
    if effort is not None and effort not in REASONING_EFFORTS:
        raise ConfigError("PMP_REASONING_EFFORT must be one of: " + ", ".join(REASONING_EFFORTS))
    token_param = _clean(env, "PMP_TOKEN_PARAM") or "max_tokens"
    if token_param not in TOKEN_PARAMS:
        raise ConfigError("PMP_TOKEN_PARAM must be one of: " + ", ".join(TOKEN_PARAMS))
    timeout = _int(env, "PMP_TIMEOUT_SECONDS", 30, 1, 600)

    if provider == "openai":
        key, base_url = openai_key, _url(env, "OPENAI_BASE_URL")
    else:
        key, base_url = groq_key, _url(env, "GROQ_BASE_URL") or GROQ_BASE_URL

    tracing = provider == "openai" and _clean(env, "PMP_TRACING").lower() in ("1", "true", "yes", "on")
    model_name = _clean(env, "PMP_MODEL") or DEFAULT_MODELS[provider]
    if effort is None and provider == "groq" and "qwen3" in model_name.lower():
        # Groq Qwen3 models support reasoning_effort="none", preventing hidden thinking tokens from exhausting OTPM.
        effort = "none"
    default_itpm = 5500 if provider == "groq" else 0
    default_otpm = 900 if provider == "groq" else 0
    default_worker_tokens = 400 if provider == "groq" else 600
    return ProviderConfig(
        provider=provider, model=model_name, api_key=key,
        base_url=base_url, structured_output=structured, timeout=float(timeout),
        max_output_tokens=_int(env, "PMP_MAX_OUTPUT_TOKENS", 1500, 64, 32000),
        worker_max_tokens=_int(env, "PMP_WORKER_MAX_TOKENS", default_worker_tokens, 64, 32000),
        max_attempts=_int(env, "PMP_LLM_MAX_ATTEMPTS", 2, 1, 4),
        provider_retries=_int(env, "PMP_PROVIDER_RETRIES", 2, 0, 3),
        itpm_limit=_int(env, "PMP_ITPM_LIMIT", default_itpm, 0, 10_000_000),
        otpm_limit=_int(env, "PMP_OTPM_LIMIT", default_otpm, 0, 10_000_000),
        reasoning_effort=effort, token_param=token_param, tracing=tracing, selection_reason=reason,
    )


def build_client(cfg: ProviderConfig):
    """Create the provider client from ``cfg`` only. No environment or SDK-global side effects."""
    if not cfg.live:
        raise ConfigError("No LLM provider is configured (offline mode). Set OPENAI_API_KEY or GROQ_API_KEY.")
    from openai import AsyncOpenAI

    # Disable SDK transport retries (max_retries=0) so application code enforces one bounded retry budget.
    kwargs: dict[str, Any] = {"api_key": cfg.api_key, "timeout": cfg.timeout, "max_retries": 0}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return AsyncOpenAI(**kwargs)


def build_model(cfg: ProviderConfig, client):
    """The SDK model object bound to ``client`` (chat-completions works on both providers)."""
    if not cfg.live:
        raise ConfigError("No LLM provider is configured (offline mode). Set OPENAI_API_KEY or GROQ_API_KEY.")
    from agents import OpenAIChatCompletionsModel

    model_name = cfg.model or DEFAULT_MODELS[cfg.provider]
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)
