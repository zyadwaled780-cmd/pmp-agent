import asyncio
import json
from datetime import date

import pytest
from agents.agent_output import AgentOutputSchema

from checks import run_all_checks
from checks.deadline_checker import DeadlineStatus, check_deadline
from checks.task_checker import check_tasks
from checks.team_checker import TeamMember, check_team
from domain.models import (
    ExecutiveReport,
    ExtractedUpdate,
    InputMessage,
    ProposedAction,
    RiskItem,
    Severity,
    SourceType,
    TaskItem,
)
from project_agents.evaluator_agent import EvalResult, evaluate_report, grounding_issues, write_and_evaluate
from project_agents.extraction_agent import ExtractionError, extract_update
from project_agents.orchestrator import OrchestrationPlan
from project_agents.risk_analyzer import RiskAnalysis, analyze_risks, deterministic_risks
from project_agents.status_writer import ReportDraft, write_report
from project_agents.workers import WorkerReport
from tools.execution_tools import ProjectStore, TransientToolError, execute_action

TODAY = date(2026, 10, 16)


def T(id, title, status="todo", owner=None, due=None, est=None, blocker=None):
    return TaskItem(id=id, title=title, status=status, owner=owner, due_date=due, estimate_days=est, blocker=blocker)


def U(**kw):
    base = dict(source=SourceType.EMPLOYEE, sender="Ahmed", summary="s", tasks=[], blockers=[], risks=[],
                mentioned_deadline=None, budget_used_percent=None)
    base.update(kw)
    return ExtractedUpdate(**base)


TASKS = [
    T("T1", "Login page", "done", "Ahmed", est=2),
    T("T2", "Payment API", "blocked", "Sara", date(2026, 10, 12), 4, "waiting for supplier"),
    T("T3", "Reports", "in_progress", "Sara", date(2026, 10, 30), 4),
    T("T4", "Docs", "todo", "Omar", date(2026, 10, 14), 1),
]
TEAM = [TeamMember(name="Ahmed"), TeamMember(name="Sara"), TeamMember(name="Omar", on_leave=True)]


def run(coro):
    return asyncio.run(coro)


# ---------- checks ----------
def test_task_checker():
    f = check_tasks(TASKS, TODAY)
    assert [t.id for t in f.overdue] == ["T2", "T4"]
    assert [t.id for t in f.blocked] == ["T2"]
    assert f.open_count == 3 and f.done_count == 1


def test_team_checker():
    f = check_team(TEAM, TASKS)
    assert f.overloaded == ["Sara"]            # 8 days of open work > 5
    assert f.unavailable == ["Omar"]
    assert f.tasks_owned_by_unavailable == ["Docs"]
    assert f.available == ["Ahmed"]


def test_team_unknown_owner():
    f = check_team(TEAM, [T("X", "t", owner="Ghost")])
    assert f.unknown_owners == ["Ghost"]


def test_deadline_at_risk_is_critical():
    f = check_deadline(TASKS, TEAM, date(2026, 10, 20), TODAY)  # 9 days work, capacity 2/day -> 4.5 > 4 days
    assert f.status == DeadlineStatus.AT_RISK and f.critical and f.slip_days == 1


def test_deadline_ok_warning_unknown_passed_nocapacity():
    assert check_deadline(TASKS, TEAM, date(2026, 12, 1), TODAY).status == DeadlineStatus.ON_TRACK
    assert check_deadline(TASKS, TEAM, date(2026, 10, 21), TODAY).status == DeadlineStatus.WARNING
    assert check_deadline(TASKS, TEAM, None, TODAY).status == DeadlineStatus.UNKNOWN
    assert check_deadline(TASKS, TEAM, date(2026, 10, 1), TODAY).critical
    assert check_deadline(TASKS, [TeamMember(name="A", on_leave=True)], date(2026, 12, 1), TODAY).critical
    assert not check_deadline([T("D", "done", "done")], TEAM, date(2026, 10, 17), TODAY).critical


def test_parallel_runner_survives_one_failure(monkeypatch):
    import checks
    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(checks, "check_team", boom)
    r = run(run_all_checks(TASKS, TEAM, date(2026, 12, 1), TODAY))
    assert r.people is None and r.tasks is not None and r.deadline is not None
    assert any("team check unavailable" in g for g in r.data_gaps)


# ---------- extraction ----------
MSG = InputMessage(source=SourceType.SUPPLIER, sender="Acme", text="Shipment will be 5 days late")


def test_extraction_forces_sender_and_drops_invented_ids():
    async def fake(name, ins, prompt, out):
        return U(source=SourceType.ACCOUNTANT, sender="HACKER", tasks=[T("T99", "x", "todo")])
    u = run(extract_update(MSG, llm=fake, known_task_ids={"T1"}))
    assert u.sender == "Acme" and u.source == SourceType.SUPPLIER and u.tasks[0].id is None


def test_extraction_reasks_once_then_succeeds_or_fails():
    calls = []
    async def flaky(name, ins, prompt, out):
        calls.append(prompt)
        if len(calls) == 1:
            raise ValueError("bad json")
        return U()
    run(extract_update(MSG, llm=flaky))
    assert len(calls) == 2 and "previous answer was invalid" in calls[1]

    async def always_bad(*a):
        raise ValueError("bad")
    with pytest.raises(ExtractionError):
        run(extract_update(MSG, llm=always_bad))


def test_extraction_wraps_message_as_data():
    seen = {}
    async def spy(name, ins, prompt, out):
        seen["p"], seen["i"] = prompt, ins
        return U()
    run(extract_update(MSG, llm=spy))
    assert "<message>" in seen["p"] and "DATA" in seen["i"]


# ---------- risks ----------
def results_for(deadline=date(2026, 10, 20)):
    return run(run_all_checks(TASKS, TEAM, deadline, TODAY))


def test_deterministic_risks():
    u = U(blockers=["Shipment 5 days late"], budget_used_percent=85)
    titles = {r.title: r.severity for r in deterministic_risks(u, results_for())}
    assert titles["Project deadline at risk"] == Severity.CRITICAL
    assert titles["Task blocked: Payment API"] == Severity.HIGH
    assert titles["Sara is overloaded"] == Severity.MEDIUM
    assert titles["Budget nearly used up"] == Severity.MEDIUM
    assert titles["Blocker reported in update"] == Severity.HIGH


def test_llm_cannot_add_invented_task_ids_or_downgrade():
    async def fake(name, ins, prompt, out):
        return RiskAnalysis(risks=[
            RiskItem(title="Bad", severity=Severity.HIGH, evidence="x", related_task_ids=["T99"]),
            RiskItem(title="Project deadline at risk", severity=Severity.LOW, evidence="dup", related_task_ids=[]),
            RiskItem(title="Supplier delay hits overloaded Sara", severity=Severity.HIGH, evidence="Sara has 8 days open", related_task_ids=["T2"]),
        ])
    risks, gaps = run(analyze_risks(U(), results_for(), llm=fake))
    titles = [r.title for r in risks]
    assert "Bad" not in titles and "Supplier delay hits overloaded Sara" in titles
    assert next(r for r in risks if r.title == "Project deadline at risk").severity == Severity.CRITICAL
    assert risks[0].severity == Severity.CRITICAL  # ranked


def test_risk_llm_failure_falls_back():
    async def boom(*a):
        raise RuntimeError("api")
    risks, gaps = run(analyze_risks(U(), results_for(), llm=boom))
    assert risks and any("rule-based risks only" in g for g in gaps)
    risks2, _ = run(analyze_risks(U(), results_for(), llm=None))
    assert risks2


# ---------- writer / evaluator ----------
def draft(summary="Payment API is blocked; deadline at risk.", actions=None):
    return ReportDraft(summary=summary, proposed_actions=actions or [])


def action(**kw):
    base = dict(action="create_project_comment", comment="Payment API blocked", reason="blocked", evidence="T2 blocked")
    base.update(kw)
    return ProposedAction(**base)


def report(summary="ok", actions=None):
    return ExecutiveReport(summary=summary, risks=[], proposed_actions=actions or [], data_gaps=[])


def test_writer_strips_forbidden_actions():
    async def fake(name, ins, prompt, out):
        return draft(actions=[action(), action(action="delete_task", task_id="T1")])
    r = run(write_report(U(), [], results_for(), [], llm=fake))
    assert [a.action for a in r.proposed_actions] == ["create_project_comment"]
    assert any("forbidden action" in g for g in r.data_gaps)


def test_grounding_rules():
    res = results_for()
    assert grounding_issues(report("Backend is 100% done"), U(), res)
    assert grounding_issues(report("We will deliver on Nov 1"), U(), res)
    assert grounding_issues(report(actions=[action(action="update_task", task_id="T99", status="in_progress")]), U(), res)
    assert grounding_issues(report(actions=[action(action="assign_task", task_id="T2", owner="Ghost")]), U(), res)
    assert grounding_issues(report(actions=[action(action="update_task", task_id="T3", status="done")]), U(), res)
    ok_update = U(tasks=[T("T3", "Reports", "done")])
    assert not grounding_issues(report(actions=[action(action="update_task", task_id="T3", status="done")]), ok_update, res)
    assert not grounding_issues(report("Payment API blocked."), U(), res)


def test_evaluator_python_overrides_lying_llm():
    async def lenient(name, ins, prompt, out):
        return EvalResult(passed=True, issues=[], feedback="ok")
    ev = run(evaluate_report(report("Backend is 100% done"), U(), results_for(), llm=lenient))
    assert not ev.passed


def test_evaluator_llm_can_fail_a_clean_report():
    async def strict(name, ins, prompt, out):
        return EvalResult(passed=False, issues=["unsupported claim"], feedback="remove it")
    ev = run(evaluate_report(report("fine"), U(), results_for(), llm=strict))
    assert not ev.passed and "unsupported claim" in ev.issues


def make_llm(drafts):
    state = {"i": 0}
    async def fake(name, ins, prompt, out):
        if out is ReportDraft:
            d = drafts[min(state["i"], len(drafts) - 1)]
            state["i"] += 1
            return d
        return EvalResult(passed=True, issues=[], feedback="ok")
    return fake


def test_loop_revises_then_approves():
    llm = make_llm([draft("Backend is 100% done"), draft("Payment API is blocked.")])
    out = run(write_and_evaluate(U(), [], results_for(), [], llm=llm))
    assert out.approved and out.iterations == 2 and not out.escalate_to_human


def test_loop_escalates_after_three():
    llm = make_llm([draft("Backend is 100% done")])
    out = run(write_and_evaluate(U(), [], results_for(), [], llm=llm))
    assert not out.approved and out.iterations == 3 and out.escalate_to_human and out.evaluation.issues


# ---------- execution ----------
def store():
    return ProjectStore([t.model_copy(deep=True) for t in TASKS])


def test_execute_happy_paths_and_readback():
    s = store()
    r = execute_action(action(action="create_task", title="New task", comment=None), s, authorized=True)
    assert r.status == "confirmed" and s.tasks[r.task_id].title == "New task"
    r = execute_action(action(action="assign_task", task_id="T3", owner="Ahmed", comment=None), s, authorized=True)
    assert r.status == "confirmed" and s.tasks["T3"].owner == "Ahmed"
    r = execute_action(action(action="update_task", task_id="T3", status="done", comment=None), s, authorized=True)
    assert s.tasks["T3"].status == "done"
    r = execute_action(action(), s, authorized=True)
    assert r.status == "confirmed" and len(s.comments) == 1


def test_execute_refusals():
    s = store()
    assert execute_action(action(action="delete_task", task_id="T1"), s, authorized=True).status == "refused"
    assert "T1" in s.tasks
    assert execute_action(action(), s, authorized=False).status == "refused"
    assert not s.comments


def test_execute_permanent_error_does_not_retry():
    r = execute_action(action(action="assign_task", task_id="NOPE", owner="A", comment=None), store(), authorized=True)
    assert r.status == "failed" and r.attempts == 1


def test_execute_retries_twice_on_transient_then_gives_up():
    class Flaky(ProjectStore):
        @property
        def comments(self):
            raise TransientToolError("timeout")
        @comments.setter
        def comments(self, v):
            pass
    f = Flaky([t.model_copy(deep=True) for t in TASKS])
    r = execute_action(action(), f, authorized=True, backoff_seconds=0)
    assert r.status == "failed" and r.attempts == 3 and "no result was invented" in r.detail


def test_execute_succeeds_on_second_attempt():
    s = store()
    state = {"n": 0}
    orig = s.new_id
    def flaky_new_id():
        state["n"] += 1
        if state["n"] == 1:
            raise TransientToolError("lock")
        return orig()
    s.new_id = flaky_new_id
    r = execute_action(action(action="create_task", title="X", comment=None), s, authorized=True, backoff_seconds=0)
    assert r.status == "confirmed" and r.attempts == 2


# ---------- schemas the model must fill ----------
@pytest.mark.parametrize("M", [ExtractedUpdate, OrchestrationPlan, WorkerReport, RiskAnalysis, ReportDraft, EvalResult])
def test_llm_output_models_are_strict_schema_safe(M):
    schema = AgentOutputSchema(M).json_schema()   # raises if the SDK rejects it
    assert '"default"' not in json.dumps(schema)


# ---------- end to end with a fake model ----------
def test_end_to_end_supplier_delay():
    async def fake(name, ins, prompt, out):
        if out is ExtractedUpdate:
            return U(source=SourceType.SUPPLIER, sender="Acme", blockers=["Shipment 5 days late"], summary="late shipment")
        if out is RiskAnalysis:
            return RiskAnalysis(risks=[])
        if out is ReportDraft:
            return draft("Deadline at risk and Payment API blocked.", [action()])
        return EvalResult(passed=True, issues=[], feedback="ok")

    async def pipeline():
        upd = await extract_update(MSG, llm=fake, known_task_ids={t.id for t in TASKS})
        res = await run_all_checks(TASKS, TEAM, date(2026, 10, 20), TODAY)
        risks, gaps = await analyze_risks(upd, res, llm=fake)
        out = await write_and_evaluate(upd, risks, res, gaps, llm=fake)
        assert res.deadline.critical and out.approved
        s = store()
        results = [execute_action(a, s, authorized=True) for a in out.report.proposed_actions]
        return results, s
    results, s = run(pipeline())
    assert results[0].status == "confirmed" and s.comments
