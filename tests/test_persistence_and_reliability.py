"""Tests for optional durable persistence, offline mocked provider failure modes, .env config, and security boundaries."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError, RateLimitError

import app as pmp_app
import pmp_config
from checks import CheckResults
from data.sample_project import build_sample_store
from domain.models import (
    ExecutiveReport,
    ExtractedUpdate,
    InputMessage,
    ProposedAction,
    SourceType,
)
from pipeline.actions import apply_action
from pipeline.adapters import Route
from pipeline.context import PMPContext
from pipeline.persistence import (
    SCHEMA_VERSION,
    InvalidSessionIdError,
    SchemaVersionError,
    SQLiteSessionStore,
    validate_session_id,
)
from pipeline.runner import run_update
from pipeline.session import (
    create_session,
    persist_session,
    reset_session,
    resolve_session_approval,
)
from project_agents import llm as llm_mod
from project_agents.claims import has_completion_claim, has_delivery_promise
from project_agents.evaluator_agent import EvalResult, grounding_findings
from project_agents.json_output import LLMOutputError
from project_agents.offline_llm import make_offline_llm
from project_agents.orchestrator import OrchestrationPlan
from project_agents.risk_analyzer import RiskAnalysis
from project_agents.status_writer import ReportDraft
from project_agents.workers import sdk_worker_runner
from tests.scripted_model import ScriptedModel, message

TODAY = date(2026, 10, 16)


def _run(coro):
    return asyncio.run(coro)


def _http_req() -> httpx.Request:
    return httpx.Request("POST", "https://api.example.invalid/v1/chat/completions")


def _http_resp(status: int) -> httpx.Response:
    return httpx.Response(status_code=status, request=_http_req())


# ===========================================================================
# Item 3 - Optional durable restart persistence
# ===========================================================================
def test_persistence_disabled_by_default_and_config_validation(tmp_path):
    default_cfg = pmp_config.load_persistence_config(env={})
    assert default_cfg.enabled is False

    enabled_cfg = pmp_config.load_persistence_config(
        env={
            "PMP_PERSIST_SESSIONS": "true",
            "PMP_SESSION_DB_PATH": str(tmp_path / "custom.db"),
        }
    )
    assert enabled_cfg.enabled is True
    assert enabled_cfg.db_path == tmp_path / "custom.db"

    with pytest.raises(pmp_config.ConfigError):
        pmp_config.load_persistence_config(env={"PMP_PERSIST_SESSIONS": "maybe"})


def test_persistence_restart_roundtrip_counter_budget_approvals_and_no_secrets(tmp_path):
    db_path = tmp_path / "sessions.db"
    backend1 = SQLiteSessionStore(db_path)

    s1 = create_session(today=TODAY, persistence=backend1)
    sid1 = s1.session_id
    llm = make_offline_llm([(t.id, t.title) for t in s1.store.tasks.values()])

    # Propose & approve update + accountant budget + create T6
    r1 = _run(
        run_update(
            InputMessage(source=SourceType.EMPLOYEE, sender="Ahmed", text="I finished the Mobile layout."),
            s1.store,
            s1.queue,
            llm=llm,
            today=s1.today,
        )
    )
    persist_session(s1)
    assert len(r1.pending_approvals) == 1
    dec = resolve_session_approval(s1, r1.pending_approvals[0], True, note="Looks good")
    assert dec.ok and dec.applied

    _run(
        run_update(
            InputMessage(source=SourceType.ACCOUNTANT, sender="Nadia", text="Spending passed 88% of the budget."),
            s1.store,
            s1.queue,
            llm=llm,
            today=s1.today,
        )
    )
    new_tid = s1.store.new_id()
    assert new_tid == "T6"
    persist_session(s1)

    # Simulate restart with a brand new SQLiteSessionStore instance
    backend2 = SQLiteSessionStore(db_path)
    restored = backend2.load_session(sid1)
    assert restored is not None
    assert restored.session_id == sid1
    assert restored.today == TODAY
    assert restored.store.tasks["T5"].status == "done"
    assert restored.store.budget_used_percent == 88.0
    assert len(restored.store.budget_history) == 1
    assert len(restored.store.comments) == 2
    # High-water mark preserved across restart: next task id is T7 (never T6 or T1)
    assert restored.store.new_id() == "T7"

    restored_items = restored.queue.all()
    assert len(restored_items) == 1
    assert restored_items[0].status == "approved"
    assert restored_items[0].decided_at is not None
    assert restored_items[0].decision_note == "Looks good"

    # Raw SQLite file contains zero keys, SDK clients, locks, or chat histories
    raw_bytes = db_path.read_text(errors="ignore")
    for forbidden in ("OPENAI_API_KEY", "GROQ_API_KEY", "sk-", "gsk_", "AsyncOpenAI", "RLock", "SYSTEM:"):
        assert forbidden not in raw_bytes


def test_persistence_session_isolation_reset_validation_and_schema_version(tmp_path):
    db_path = tmp_path / "iso.db"
    backend = SQLiteSessionStore(db_path)

    s1 = create_session(today=TODAY, persistence=backend)
    s2 = create_session(today=TODAY, persistence=backend)
    s1_old_id = s1.session_id
    s2_id = s2.session_id

    s1.store.comments.append({"task_id": None, "text": "session-1-only"})
    s2.store.comments.append({"task_id": None, "text": "session-2-only"})
    s2.store.set_budget(91.0, source="Nadia")
    persist_session(s1)
    persist_session(s2)

    # Resetting s1 deletes s1's prior session record & creates fresh s1 without touching s2
    new_day = TODAY + timedelta(days=5)
    reset_session(s1, today=new_day)
    assert s1.session_id != s1_old_id
    assert backend.load_session(s1_old_id) is None

    reloaded_s2 = backend.load_session(s2_id)
    assert reloaded_s2 is not None
    assert reloaded_s2.store.budget_used_percent == 91.0
    assert reloaded_s2.store.comments == [{"task_id": None, "text": "session-2-only"}]

    # Opaque session_id validation rejects path traversal and invalid formats
    for bad_id in ("", "../etc/passwd", "..\\win", "short", "G" * 32, "a" * 31 + "/"):
        with pytest.raises(InvalidSessionIdError):
            validate_session_id(bad_id)
        with pytest.raises(InvalidSessionIdError):
            backend.load_session(bad_id)

    # Schema version mismatch must fail cleanly
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE schema_version SET version = 999")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaVersionError):
        SQLiteSessionStore(db_path)
    assert SCHEMA_VERSION == 1


def test_persistence_interrupted_execution_recovery_and_restart_revalidation(tmp_path):
    db_path = tmp_path / "recovery.db"
    backend = SQLiteSessionStore(db_path)
    sess = create_session(today=TODAY, persistence=backend)

    pid = sess.queue.add(
        ProposedAction(action="update_task", task_id="T5", status="done", reason="done", evidence="Finished"),
        Route.NORMAL,
        "requires review",
    )
    persist_session(sess)

    # Simulate crash after mark_executing was committed to SQLite before completion commit
    backend.mark_executing(sess.session_id, pid)

    recovered_backend = SQLiteSessionStore(db_path)
    restored = recovered_backend.load_session(sess.session_id)
    assert restored is not None
    item = restored.queue.get(pid)
    assert item is not None
    assert item.status == "interrupted"

    # Attempting to approve an interrupted item must NOT silently re-execute it
    res = resolve_session_approval(restored, pid, True)
    assert res.ok and not res.applied
    assert "interrupted" in res.message
    assert restored.store.tasks["T5"].status == "in_progress"


def test_persistence_revalidates_queued_action_against_updated_store_after_restart(tmp_path):
    db_path = tmp_path / "reval.db"
    backend = SQLiteSessionStore(db_path)
    sess = create_session(today=TODAY, persistence=backend)

    # Queue update_task setting T5 to done (valid while T5 is in_progress)
    pid = sess.queue.add(
        ProposedAction(
            action="update_task",
            task_id="T5",
            status="done",
            reason="completed",
            evidence="Finished Mobile layout",
        ),
        Route.NORMAL,
        "status change",
    )
    # Before restart approval, T5 already transitioned to 'done' via another action/state change
    sess.store.tasks["T5"].status = "done"
    persist_session(sess)

    restored = SQLiteSessionStore(db_path).load_session(sess.session_id)
    assert restored is not None
    outcome = resolve_session_approval(restored, pid, True)
    assert outcome.ok and outcome.applied
    assert outcome.item is not None and outcome.item.status == "failed"
    assert "redundant" in outcome.message.lower() or "already" in outcome.message.lower()


# ===========================================================================
# Item 4 - .env config behavior & offline mocked live-failure coverage
# ===========================================================================
def test_dotenv_loading_without_mutating_os_environ_and_placeholder_safety(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("PMP_PROVIDER", raising=False)

    # Copying .env.example (placeholders/commented) stays safely offline
    dotenv_example_copy = tmp_path / ".env.placeholder"
    dotenv_example_copy.write_text(
        "PMP_PROVIDER=auto\nOPENAI_API_KEY=your-openai-api-key-here\nGROQ_API_KEY=your-groq-api-key-here\n",
        encoding="utf-8",
    )
    cfg_placeholder = pmp_config.load_config(dotenv_path=dotenv_example_copy)
    assert not cfg_placeholder.live and cfg_placeholder.provider == "offline"

    # Real .env key loads into config without modifying os.environ
    dotenv_live = tmp_path / ".env"
    dotenv_live.write_text("GROQ_API_KEY=gsk_dotenv_fake_token_123456\nPMP_PROVIDER=groq\n", encoding="utf-8")
    cfg_dotenv = pmp_config.load_config(dotenv_path=dotenv_live)
    assert cfg_dotenv.live and cfg_dotenv.provider == "groq"
    assert cfg_dotenv.api_key == "gsk_dotenv_fake_token_123456"
    assert "GROQ_API_KEY" not in os.environ

    # Real process env overrides .env cleanly
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proc-env-token-987654")
    monkeypatch.setenv("PMP_PROVIDER", "openai")
    cfg_override = pmp_config.load_config(dotenv_path=dotenv_live)
    assert cfg_override.provider == "openai"
    assert cfg_override.api_key == "sk-proc-env-token-987654"


@pytest.mark.parametrize(
    ("forced", "env_keys"),
    [
        ("openai", {"GROQ_API_KEY": "gsk_only_groq_key_123456"}),
        ("groq", {"OPENAI_API_KEY": "sk-only-openai-key-123456"}),
    ],
)
def test_forced_provider_without_matching_key_never_switches(forced, env_keys):
    with pytest.raises(pmp_config.ConfigError):
        pmp_config.load_config({"PMP_PROVIDER": forced, **env_keys})


@pytest.mark.parametrize(
    ("exc_factory", "expected_calls", "expected_hint"),
    [
        (
            lambda secret: AuthenticationError(f"Invalid API key {secret}", response=_http_resp(401), body=None),
            1,
            "Authentication failed",
        ),
        (
            lambda secret: RateLimitError(f"Rate limit hit for {secret}", response=_http_resp(429), body=None),
            2,
            "rate limit",
        ),
        (
            lambda _secret: APITimeoutError(request=_http_req()),
            2,
            "timed out",
        ),
        (
            lambda secret: APIConnectionError(message=f"Connection failed ({secret})", request=_http_req()),
            2,
            "Could not connect",
        ),
    ],
)
def test_provider_errors_redacted_never_leak_secrets_and_never_switch_provider(
    exc_factory, expected_calls, expected_hint
):
    llm_mod.reset_runtime()
    secret = "sk-live-secret-token-abcdef12345"
    sleeps: list[float] = []

    async def fake_sleep(delay: float):
        sleeps.append(delay)

    cfg = pmp_config.load_config(
        {"PMP_PROVIDER": "openai", "OPENAI_API_KEY": secret, "PMP_PROVIDER_RETRIES": "1"}
    )
    rt = llm_mod.configure_runtime(cfg, client=SimpleNamespace(), sleep_fn=fake_sleep)
    calls = {"n": 0}

    def failing_model(instructions, prompt, tools):
        calls["n"] += 1
        raise exc_factory(secret)

    # Drive real Runner path via _call_once model attribute while preserving live Runtime path
    rt.model = ScriptedModel(failing_model)
    try:
        store = build_sample_store(TODAY)
        queue = pmp_app.ApprovalQueue()
        snapshot_tasks = {k: t.model_copy(deep=True) for k, t in store.tasks.items()}
        res = _run(
            run_update(
                InputMessage(source=SourceType.EMPLOYEE, sender="Ahmed", text="Status update on Mobile layout."),
                store,
                queue,
                llm=llm_mod.call_llm,
                today=TODAY,
            )
        )
        assert calls["n"] == expected_calls
        assert len(sleeps) == expected_calls - 1
        assert res.status == "error"
        assert expected_hint.lower() in res.message.lower()
        assert "could not understand" not in res.message.lower()
        assert res.executions == [] and res.pending_approvals == []
        assert store.tasks == snapshot_tasks and not store.comments and not queue.all()
        assert all(secret not in (step.summary or "") and secret not in (step.output or "") for step in res.steps)
        assert secret not in res.message
        assert any("provider call failed" in ev for ev in rt.events)
        assert all(secret not in ev for ev in rt.events)
    finally:
        llm_mod.reset_runtime()


def test_transient_provider_retry_recovers_respecting_retry_after():
    llm_mod.reset_runtime()
    sleeps: list[float] = []

    async def fake_sleep(delay: float):
        sleeps.append(delay)

    cfg = pmp_config.load_config(
        {"PMP_PROVIDER": "groq", "GROQ_API_KEY": "gsk_fake_retry_token_123456", "PMP_PROVIDER_RETRIES": "2"}
    )
    rt = llm_mod.configure_runtime(cfg, client=SimpleNamespace(), sleep_fn=fake_sleep)
    calls = {"n": 0}
    resp_429 = httpx.Response(429, headers={"retry-after": "1.5"}, request=_http_req())

    def transient_then_ok(instructions, prompt, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("Rate limit reached. Please try again in 1.2s.", response=resp_429, body=None)
        return [
            message(
                json.dumps(
                    {
                        "source": "employee",
                        "sender": "Ahmed",
                        "summary": "Ahmed finished Mobile layout",
                        "tasks": [],
                        "blockers": [],
                        "risks": [],
                        "mentioned_deadline": None,
                        "budget_used_percent": None,
                    }
                )
            )
        ]

    rt.model = ScriptedModel(transient_then_ok)
    try:
        out = _run(llm_mod.call_llm("Extraction Agent", "Extract", "hi", ExtractedUpdate))
        assert isinstance(out, ExtractedUpdate)
        assert calls["n"] == 2
        assert len(sleeps) == 1 and sleeps[0] == pytest.approx(1.75)
        assert any("retrying 1/2" in ev for ev in rt.events)
    finally:
        llm_mod.reset_runtime()


def test_groq_tool_worker_avoids_json_object_mode_while_non_tool_calls_keep_it():
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage

    llm_mod.reset_runtime()
    seen_settings = []

    class _InspectModel(Model):
        async def get_response(
            self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, **kw
        ):
            seen_settings.append((bool(tools), model_settings.extra_args))
            if tools:
                return ModelResponse(
                    output=[message('{"findings": ["2 overdue"], "concerns": ["T2 blocked"]}')],
                    usage=Usage(),
                    response_id=None,
                )
            return ModelResponse(
                output=[
                    message(
                        json.dumps(
                            {
                                "source": "employee",
                                "sender": "Ahmed",
                                "summary": "Checked status",
                                "tasks": [],
                                "blockers": [],
                                "risks": [],
                                "mentioned_deadline": None,
                                "budget_used_percent": None,
                            }
                        )
                    )
                ],
                usage=Usage(),
                response_id=None,
            )

        async def stream_response(self, *a, **k):
            raise NotImplementedError
            yield  # pragma: no cover

    cfg = pmp_config.load_config({"PMP_PROVIDER": "groq", "GROQ_API_KEY": "gsk_fake_groq_token_123456"})
    rt = llm_mod.configure_runtime(cfg, client=SimpleNamespace())
    rt.model = _InspectModel()
    ctx = PMPContext(store=build_sample_store(TODAY), queue=pmp_app.ApprovalQueue(), today=TODAY)
    try:
        rep = _run(sdk_worker_runner("tasks", ctx))
        assert rep.findings == ["2 overdue"]
        # Tool worker must NOT attach response_format={"type": "json_object"}
        assert seen_settings[0][0] is True
        assert not (seen_settings[0][1] or {}).get("response_format")

        # Non-tool calls must still retain json_object structured output
        _run(llm_mod.call_llm("Extraction Agent", "Extract", "hi", ExtractedUpdate))
        assert seen_settings[1][0] is False
        assert (seen_settings[1][1] or {}).get("response_format") == {"type": "json_object"}
        assert rt.structured_mode == "json_object"
    finally:
        llm_mod.reset_runtime()


def test_app_decide_prevents_duplicate_execution_and_build_client_disables_sdk_retries():
    cfg = pmp_config.load_config({"PMP_PROVIDER": "groq", "GROQ_API_KEY": "gsk_fake_groq_token_123456"})
    client = pmp_config.build_client(cfg)
    try:
        assert client.max_retries == 0
    finally:
        _run(client.close())

    sess = create_session(today=TODAY)
    res = _run(
        run_update(
            InputMessage(source=SourceType.EMPLOYEE, sender="Ahmed", text="I finished the Mobile layout."),
            sess.store,
            sess.queue,
            llm=make_offline_llm([(t.id, t.title) for t in sess.store.tasks.values()]),
            today=TODAY,
        )
    )
    pid = res.pending_approvals[0]
    note1, *_ = _run(pmp_app.decide(pid, True, sess))
    note2, *_ = _run(pmp_app.decide(pid, True, sess))
    assert "approved:" in note1
    assert "Already decided" in note2
    assert sess.store.tasks["T5"].status == "done"


def test_token_quota_pacer_enforces_separate_itpm_and_otpm_windows():
    now = {"t": 100.0}
    slept: list[float] = []

    async def fake_sleep(delay: float):
        slept.append(delay)
        now["t"] += delay

    pacer = llm_mod.TokenQuotaPacer(itpm_limit=5000, otpm_limit=500, clock_fn=lambda: now["t"])
    pacer.record(1000, 450)
    # Fits ITPM (1000+500 <= 5000) but exceeds OTPM (450+100 > 500) -> waits ~60.1s? Wait capped at 25s
    waited = _run(pacer.pace(500, 100, fake_sleep))
    assert waited > 0 and slept == [waited]

    # Advance past 60s window -> zero wait needed
    now["t"] += 61.0
    assert _run(pacer.pace(500, 100, fake_sleep)) == 0.0

    # Test Groq Qwen3 default reasoning_effort='none' & explicit override
    cfg_qwen = pmp_config.load_config({"PMP_PROVIDER": "groq", "GROQ_API_KEY": "gsk_fake_groq_token_123456"})
    assert cfg_qwen.reasoning_effort == "none"
    assert cfg_qwen.itpm_limit == 5500 and cfg_qwen.otpm_limit == 900
    cfg_custom = pmp_config.load_config(
        {"PMP_PROVIDER": "groq", "GROQ_API_KEY": "gsk_fake_groq_token_123456", "PMP_REASONING_EFFORT": "low"}
    )
    assert cfg_custom.reasoning_effort == "low"

    ms = llm_mod.make_model_settings(cfg_qwen, 600, "prompt", is_tool_agent=True)
    assert ms.temperature == 0.0 and ms.parallel_tool_calls is False


def test_provider_wait_does_not_block_decide_or_reset_and_discards_stale_turn():
    sess = create_session(today=TODAY)
    pid = sess.queue.add(
        ProposedAction(action="update_task", task_id="T5", status="done", reason="done", evidence="done"),
        Route.NORMAL,
        "requires review",
    )
    entered_llm = asyncio.Event()
    release_llm = asyncio.Event()

    def slow_llm_factory(_tasks):
        async def _llm(name, instructions, prompt, output_type):
            entered_llm.set()
            await release_llm.wait()
            return ExtractedUpdate(
                source=SourceType.EMPLOYEE,
                sender="Ahmed",
                summary="Slow update",
                tasks=[],
                blockers=[],
                risks=[],
            )

        return _llm

    async def scenario():
        run_task = asyncio.create_task(
            pmp_app.on_run(
                "employee", "Ahmed", "Status check", "English", True, sess, llm_factory=slow_llm_factory
            )
        )
        await entered_llm.wait()

        # Manager decide() and reset() must NOT block on the in-flight provider wait!
        note, *_ = await asyncio.wait_for(pmp_app.decide(pid, True, sess), timeout=1.0)
        assert "approved:" in note
        assert sess.store.tasks["T5"].status == "done"

        reset_note, *_ = await asyncio.wait_for(pmp_app.reset(sess), timeout=1.0)
        assert "reset" in reset_note.lower()
        assert sess.store.tasks["T5"].status == "in_progress"

        release_llm.set()
        banner, *_ = await run_task
        assert "Session was reset" in banner
        # Post-reset store remains clean (stale turn did not add comments)
        assert sess.store.comments == []

    _run(scenario())


def test_malformed_response_retries_and_structured_output_fallback_stays_on_same_provider():
    llm_mod.reset_runtime()
    secret = "gsk_groq_secret_token_987654321"
    cfg = pmp_config.load_config(
        {
            "PMP_PROVIDER": "groq",
            "GROQ_API_KEY": secret,
            "PMP_STRUCTURED_OUTPUT": "json_schema",
            "PMP_LLM_MAX_ATTEMPTS": "2",
        }
    )
    rt = llm_mod.configure_runtime(cfg, client=SimpleNamespace())
    calls = {"n": 0}

    def scripted_fallback(instructions, prompt, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BadRequestError("json_schema response_format unsupported", response=_http_resp(400), body=None)
        if calls["n"] == 2:
            return [message("not valid json at all <<< broken")]
        return [
            message(
                json.dumps(
                    {
                        "source": "employee",
                        "sender": "Ahmed",
                        "summary": "Ahmed finished Mobile layout",
                        "tasks": [],
                        "blockers": [],
                        "risks": [],
                        "mentioned_deadline": None,
                        "budget_used_percent": None,
                    }
                )
            )
        ]

    rt.model = ScriptedModel(scripted_fallback)
    try:
        out = _run(llm_mod.call_llm("Extractor", "Extract", "Ahmed finished Mobile layout", ExtractedUpdate))
        assert isinstance(out, ExtractedUpdate)
        assert calls["n"] == 3
        assert rt.structured_mode == "prompt"
        assert rt.config.provider == "groq"  # never switched provider
        assert any("falling back to prompt-only JSON" in ev for ev in rt.events)

        # Exhausting malformed retries raises LLMOutputError with redacted diagnostics
        calls2 = {"n": 0}
        rt.model = ScriptedModel(lambda *_: (calls2.__setitem__("n", calls2["n"] + 1) or [message(f"bad {secret}")]))
        with pytest.raises(LLMOutputError) as exc_info:
            _run(llm_mod.call_llm("Extractor", "Extract", "hello", ExtractedUpdate))
        assert calls2["n"] == 2
        assert secret not in str(exc_info.value)
    finally:
        llm_mod.reset_runtime()


# ===========================================================================
# Item 6 - Strengthened concurrency, guardrails, and bilingual edge cases
# ===========================================================================
def test_concurrent_session_runs_and_competing_approvals_on_persisted_store(tmp_path):
    db_path = tmp_path / "concurrent.db"
    backend = SQLiteSessionStore(db_path)

    s1 = create_session(today=TODAY, persistence=backend)
    s2 = create_session(today=TODAY, persistence=backend)

    async def run_both():
        await asyncio.gather(
            pmp_app.on_run("employee", "Ahmed", "I finished the Mobile layout.", "English", True, s1),
            pmp_app.on_run("accountant", "Nadia", "Spending passed 85% of the budget.", "English", True, s2),
        )

    _run(run_both())
    assert len(s1.queue.pending()) == 1
    assert len(s2.queue.pending()) == 0
    assert s2.store.budget_used_percent == 85.0
    assert s1.store.budget_used_percent is None

    # Race 10 threads trying to approve/reject the exact same approval id on s1
    pid = s1.queue.pending()[0].id
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(resolve_session_approval, s1, pid, i % 2 == 0) for i in range(10)]
        outcomes = [f.result() for f in futures]

    assert sum(1 for o in outcomes if o.applied) == 1
    assert all(o.ok for o in outcomes)

    reloaded = SQLiteSessionStore(db_path).load_session(s1.session_id)
    assert reloaded is not None
    assert reloaded.queue.all()[0].status in ("approved", "rejected")


def test_guardrail_failure_keeps_only_safe_actions_and_executes_nothing():
    store = build_sample_store(TODAY)
    queue = pmp_app.ApprovalQueue()

    async def unsafe_llm(name, instructions, prompt, output_type):
        if output_type is ExtractedUpdate:
            return ExtractedUpdate(
                source=SourceType.EMPLOYEE,
                sender="Ahmed",
                summary="Checked server metrics",
                tasks=[],
                blockers=[],
                risks=[],
            )
        if output_type is OrchestrationPlan:
            return OrchestrationPlan(workers=["tasks"], focus="Check metrics")
        if output_type is RiskAnalysis:
            return RiskAnalysis(risks=[], data_gaps=[])
        if output_type is EvalResult:
            return EvalResult(passed=True, issues=[])
        if output_type is ReportDraft:
            return ReportDraft(
                summary="Task T1 is completed and we guarantee Friday delivery.",
                risks=[],
                proposed_actions=[
                    ProposedAction(
                        action="create_project_comment",
                        comment="Checked server metrics today.",
                        reason="safe log",
                        evidence="Checked server metrics",
                    ),
                    ProposedAction(
                        action="create_project_comment",
                        comment="We guarantee shipping on Friday.",
                        reason="unsafe promise in comment",
                        evidence="fabricated",
                    ),
                ],
                data_gaps=[],
            )
        raise AssertionError(output_type)

    res = _run(
        run_update(
            InputMessage(source=SourceType.EMPLOYEE, sender="Ahmed", text="Checked server metrics"),
            store,
            queue,
            llm=unsafe_llm,
            today=TODAY,
        )
    )
    assert res.status == "needs_manager"
    assert res.executions == []  # NOTHING executed automatically when output guardrail fails
    assert store.comments == []
    assert res.report is not None and res.report.review_status == "withheld"
    # Unsafe comment was filtered out; only the safe comment remains queued for manager decision
    assert len(res.pending_approvals) == 1
    queued_item = queue.get(res.pending_approvals[0])
    assert queued_item is not None
    assert queued_item.action.comment == "Checked server metrics today."


def test_english_and_arabic_grounding_and_negation_edge_cases():
    # Negated / uncompleted statements must NOT trigger false completion claims
    assert not has_completion_claim("Not all tasks are done yet")
    assert not has_completion_claim("The project is not complete yet")
    assert not has_completion_claim("لم يكتمل المشروع بعد")
    assert not has_completion_claim("المهام مش خلصانة كلها لسه")

    # Affirmative project-wide completion overclaims & delivery promises MUST be detected
    assert has_completion_claim("All tasks are complete.")
    assert has_completion_claim("Everything is finished.")
    assert has_completion_claim("تم الانتهاء من كل المهام بنجاح")
    assert has_completion_claim("المشروع اكتمل بالكامل")
    assert has_delivery_promise("We will ship by Friday.")
    assert has_delivery_promise("سنسلم المشروع غداً")
    assert not has_delivery_promise("There is risk we may miss Friday delivery.")

    # Single-task completion action without update evidence is caught by grounding_findings
    store = build_sample_store(TODAY)
    findings = grounding_findings(
        ExecutiveReport(
            summary="Checked server logs.",
            risks=[],
            proposed_actions=[
                ProposedAction(
                    action="update_task",
                    task_id="T5",
                    status="done",
                    reason="ungrounded done",
                    evidence="none",
                )
            ],
        ),
        ExtractedUpdate(
            source=SourceType.EMPLOYEE,
            sender="Ahmed",
            summary="Checked server logs.",
            tasks=[],
            blockers=[],
            risks=[],
        ),
        CheckResults(),
        tasks=list(store.tasks.values()),
    )
    assert findings


def test_apply_action_blocks_invalid_or_on_leave_actions():
    store = build_sample_store(TODAY)
    ctx = PMPContext(store=store, queue=pmp_app.ApprovalQueue(), today=TODAY, route=Route.NORMAL)

    # Assigning/creating for an on-leave team member ("Omar") or unknown task is blocked cleanly
    bad_assign = ProposedAction(
        action="assign_task",
        task_id="T5",
        owner="Omar",
        reason="reassign",
        evidence="assign to Omar",
    )
    assert apply_action(bad_assign, ctx).status == "blocked"
    assert store.tasks["T5"].owner == "Ahmed"

    bad_task = ProposedAction(
        action="update_task",
        task_id="T99",
        status="done",
        reason="unknown",
        evidence="T99 done",
    )
    assert apply_action(bad_task, ctx).status == "blocked"
