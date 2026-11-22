"""The whole workflow, start to finish, with a step-by-step trace for the UI.

message -> input guardrail -> extract -> route (code) -> load data + safety scan -> orchestrator
-> [parallel: worker agents + deterministic checks] -> deadline rules -> risk analysis
-> writer/evaluator loop -> output guardrail -> action gate (+ validation) -> approval queue / execution

Both the fast path (low-signal greetings with normal routing) and the normal path share deterministic
routing, safety scans, shared ``deterministic_risks``, and the ``_synthesize_and_execute`` guardrail /
approval gate stage.
"""
import asyncio
import re
import time
from contextlib import contextmanager, nullcontext
from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from checks import run_all_checks
from checks.budget_checker import validate_budget_percent
from domain.models import ExecutiveReport, ExtractedUpdate, InputMessage, RiskItem, SourceType
from pipeline.actions import apply_action
from pipeline.adapters import (
    Route,
    check_output_text,
    clean_sender,
    input_guardrail,
    output_guardrail,
    route_update,
    scan_comments,
    scan_members,
    scan_tasks,
)
from pipeline.approvals import ApprovalQueue
from pipeline.context import PMPContext
from pmp_config import ConfigError, redact
from project_agents.evaluator_agent import write_and_evaluate
from project_agents.extraction_agent import ExtractionError, extract_update
from project_agents.llm import (
    GuardrailTripped,
    LLMCall,
    ProviderCallError,
    call_llm,
    get_runtime,
    get_settings,
    run_settings,
)
from project_agents.orchestrator import OrchestrationPlan, plan_work
from project_agents.risk_analyzer import _rank, analyze_risks_detailed, deterministic_risks
from project_agents.status_writer import write_report
from project_agents.workers import WorkerReport, WorkerRunner, rules_worker_runner, run_worker_agents
from tools.execution_tools import ExecutionResult, ProjectStore
from tools.read_tools import (
    get_project_budget,
    get_project_comments,
    get_project_deadline,
    get_project_tasks,
    get_team_members,
)

GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|greetings|welcome|good\s*(morning|afternoon|evening)|howdy|sup|status|project\s*status|"
    r"اهلا|أهلا|مرحبا|سلام|صباح\s*الخير|مساء\s*الخير)\b",
    re.I,
)
QUESTION_WORDS = re.compile(
    r"\b(who|overload\w*|busy|workload|when|why|what|which|deadline|budget|blocker|risk\w*)\b|"
    r"مشغول|ضغط|مين|امتى|متى|ميزانية",
    re.I,
)

_TRACING_DISABLED = False


class StepRecord(BaseModel):
    name: str
    kind: Literal["code", "llm", "guard", "human", "tool"]
    status: Literal["ok", "warning", "blocked", "skipped", "error"]
    summary: str
    output: Optional[str] = None
    ms: int = 0


class RunResult(BaseModel):
    status: Literal["completed", "needs_manager", "quarantined", "refused", "needs_clarification", "error"]
    message: str
    route: Optional[Route] = None
    update: Optional[ExtractedUpdate] = None
    plan: Optional[OrchestrationPlan] = None
    report: Optional[ExecutiveReport] = None
    iterations: int = 0
    executions: list[ExecutionResult] = Field(default_factory=list)
    pending_approvals: list[str] = Field(default_factory=list)
    blocked_actions: list[str] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    worker_reports: dict[str, WorkerReport] = Field(default_factory=dict)
    tool_audit: list[str] = Field(default_factory=list)


@contextmanager
def workflow_trace(name: str):
    """Enable an OpenAI trace ONLY when live tracing is explicitly turned on."""
    cm: Any = nullcontext()
    if not _TRACING_DISABLED and get_settings().model is None:
        try:
            if get_runtime().config.tracing:
                from agents import trace
                cm = trace(name)
        except Exception:
            cm = nullcontext()
    with cm:
        yield


def disable_tracing() -> None:
    global _TRACING_DISABLED
    _TRACING_DISABLED = True


def _result_bits(results) -> list[str]:
    bits = []
    if results.tasks:
        bits.append(f"tasks: {len(results.tasks.overdue)} overdue, {len(results.tasks.blocked)} blocked")
    if results.people:
        bits.append(f"people: {len(results.people.overloaded)} overloaded, {len(results.people.unavailable)} on leave")
    if results.deadline:
        bits.append(f"deadline: {results.deadline.status.value}")
    if results.budget:
        bits.append(f"budget: {results.budget.status.value}")
    return bits


async def run_update(
    message: InputMessage,
    store: ProjectStore,
    approvals: ApprovalQueue,
    llm: LLMCall = call_llm,
    today: Optional[date] = None,
    language: str = "English",
    session=None,                              # accepted for backward compatibility; extraction stays stateless
    model=None,                                # model name or SDK Model (tests use a scripted one)
    worker_runner: Optional[WorkerRunner] = None,   # sdk_worker_runner for real worker agents
    user_role: str = "manager",
) -> RunResult:
    ctx = PMPContext(store=store, queue=approvals, today=today or date.today(), user_role=user_role)
    with run_settings(model=model, session=None, context=ctx), workflow_trace("PMP project update"):
        return await _run(message, ctx, llm, language, worker_runner)


async def _run(message: InputMessage, ctx: PMPContext, llm, language, worker_runner) -> RunResult:
    store, today = ctx.store, ctx.today
    steps: list[StepRecord] = []
    alerts: list[str] = []
    clock = {"t": time.perf_counter()}

    def rec(name, kind, status, summary, output=None):
        now = time.perf_counter()
        ms = int((now - clock["t"]) * 1000)
        steps.append(StepRecord(name=name, kind=kind, status=status, summary=summary, output=output, ms=ms))
        clock["t"] = now

    def finish(status, msg, **kw) -> RunResult:
        return RunResult(status=status, message=msg, steps=steps, alerts=alerts, tool_audit=list(ctx.audit), **kw)

    async def _synthesize_and_execute(
        update: ExtractedUpdate,
        route: Route,
        plan: OrchestrationPlan,
        tasks,
        results,
        worker_reports,
        worker_notes,
        risks: list[RiskItem],
        gaps: list[str],
        model_titles: set[str],
        *,
        allow_actions: bool,
        evaluator_llm,
    ) -> RunResult:
        try:
            outcome = await write_and_evaluate(
                update, risks, results, gaps,
                llm=llm,
                language=language,
                notes=worker_notes,
                model_risk_titles=model_titles,
                evaluator_llm=evaluator_llm,
                tasks=tasks,
                allow_actions=allow_actions,
            )
        except GuardrailTripped as exc:
            rec("Write + review", "guard", "blocked", f"Output guardrail stopped the writer twice: {exc.info}")
            return finish("needs_manager", "The report could not be produced safely. The manager must review manually.",
                          route=route, update=update, plan=plan, worker_reports=worker_reports)

        report = outcome.report
        rec("Write + review", "llm", "ok" if outcome.approved else "warning",
            f"{outcome.iterations} round(s); " + ("approved by the evaluator" if outcome.approved
                                                  else f"NOT approved ({outcome.termination_reason}), sent to the manager: "
                                                       + "; ".join(outcome.evaluation.issues)),
            report.model_dump_json(indent=1))

        force_review = not outcome.approved
        og = output_guardrail(report)
        if not og.ok:
            try:
                report = await write_report(
                    update, risks, results, gaps,
                    feedback=og.reason, previous=report,
                    llm=llm, language=language, notes=worker_notes,
                    allow_actions=allow_actions, model_risk_titles=model_titles,
                )
            except GuardrailTripped as exc:
                safe_report = report.model_copy(update={
                    "summary": "The draft report was withheld by the output guardrail and sent for manager review.",
                    "proposed_actions": [],
                    "review_status": "withheld",
                    "data_gaps": [*report.data_gaps, f"Output guardrail: {exc.info}"],
                })
                rec("Output guardrail", "guard", "blocked", f"Output guardrail stopped retry: {exc.info}")
                return finish("needs_manager", "The report was withheld for manager review. Nothing was executed.",
                              route=route, update=update, plan=plan, report=safe_report,
                              iterations=outcome.iterations, worker_reports=worker_reports)
            og = output_guardrail(report)

        if og.ok:
            rec("Output guardrail", "guard", "ok", "No promises or unproven completion claims")
        else:
            force_review = True
            safe_actions = [a for a in report.proposed_actions if check_output_text(a.comment or "").ok]
            report = report.model_copy(update={
                "summary": "The draft report was withheld by the output guardrail and sent for manager review.",
                "proposed_actions": safe_actions,
                "review_status": "withheld",
                "data_gaps": [*report.data_gaps, f"Output guardrail: {og.reason}"],
            })
            rec("Output guardrail", "guard", "blocked", og.reason + " (withheld and sent to the manager)")

        executions: list[ExecutionResult] = []
        pending: list[str] = []
        blocked: list[str] = []
        for a in report.proposed_actions:
            label = a.action + (f" {a.task_id}" if a.task_id else "")
            o = apply_action(a, ctx, force_review=force_review)
            if o.status == "blocked":
                blocked.append(f"{label}: {o.detail}")
            elif o.status == "queued":
                if o.approval_id is not None:
                    pending.append(o.approval_id)
            elif o.execution:
                executions.append(o.execution)

        rec("Action gate", "code", "warning" if (pending or blocked) else "ok",
            f"{len(report.proposed_actions)} proposed: {len(executions)} ran, {len(pending)} waiting for the manager, {len(blocked)} blocked")
        if executions:
            bad = [e for e in executions if e.status != "confirmed"]
            rec("Execution", "tool", "warning" if bad else "ok",
                "; ".join(f"{e.action}: {e.status} ({e.detail})" for e in executions))

        if pending or force_review:
            msg = f"Report ready. {len(pending)} action(s) are waiting for the manager."
            status = "needs_manager"
        else:
            msg = "Done. The report is ready and the allowed actions were executed."
            status = "completed"
        return finish(status, msg, route=route, update=update, plan=plan, report=report,
                      iterations=outcome.iterations, executions=executions,
                      pending_approvals=pending, blocked_actions=blocked, worker_reports=worker_reports)

    try:
        cleaned_sender, sender_flagged = clean_sender(message.sender)
        if sender_flagged:
            alerts.append("Hidden instruction removed from sender name.")
        clean_msg = message.model_copy(update={"sender": cleaned_sender})

        rec("Intake", "code", "ok",
            f"{clean_msg.source.value} message from {clean_msg.sender} ({len(clean_msg.text)} characters)")

        g = input_guardrail(clean_msg.text, clean_msg.source.value, clean_msg.sender)
        if not g.ok:
            rec("Input guardrail", "guard", "blocked", g.reason)
            if g.category == "injection":
                alerts.append(f"Manager alert: a suspicious message from {clean_msg.sender} was quarantined.")
                return finish("quarantined", "Message blocked: it tries to override the system's rules. Nothing was executed.")
            if g.category == "destructive":
                return finish("refused", "Refused: bulk delete requests are not allowed. Nothing was executed.")
            return finish("needs_clarification", "The message is empty. Please send an update.")
        rec("Input guardrail", "guard", "ok", "No injection or destructive request found")

        try:
            update = await extract_update(clean_msg, llm=llm, today=today, known_task_ids=set(store.tasks))
        except GuardrailTripped as exc:
            rec("Extraction", "guard", "blocked", f"SDK input guardrail stopped the agent: {exc.info}")
            alerts.append(f"Manager alert: a suspicious message from {clean_msg.sender} was quarantined.")
            return finish("quarantined", "Message blocked by the agent's input guardrail. Nothing was executed.")
        except ProviderCallError as exc:
            safe_msg = redact(exc.user_message)
            rec("Extraction", "llm", "error", safe_msg)
            return finish("error", safe_msg)
        except ExtractionError as exc:
            rec("Extraction", "llm", "error", redact(exc))
            return finish("needs_clarification", "I could not understand this message. Please rephrase it or add details.")
        rec("Extraction", "llm", "ok",
            f"{len(update.tasks)} task(s), {len(update.blockers)} blocker(s), {len(update.risks)} risk(s)",
            update.model_dump_json(indent=1))

        # Validate and persist budget updates reported by an accountant
        extra_gaps: list[str] = []
        if update.budget_used_percent is not None:
            try:
                pct = validate_budget_percent(update.budget_used_percent)
                update.budget_used_percent = pct
                if update.source == SourceType.ACCOUNTANT:
                    store.set_budget(pct, source=update.sender)
            except ValueError as exc:
                extra_gaps.append(f"Invalid budget value ignored ({exc})")
                update.budget_used_percent = None

        # ALWAYS run critical routing checks BEFORE any low-signal shortcut
        decision = route_update(update, clean_msg.text)
        route = ctx.route = decision.route
        rec("Router", "code", "warning" if route == Route.CRITICAL_PATH else "ok",
            ("CRITICAL PATH: " if route == Route.CRITICAL_PATH else "Normal path: ") + decision.reason)
        if route == Route.CRITICAL_PATH:
            alerts.append("Critical path: the manager is alerted and every change to a task needs approval.")

        # Load & sanitize project state copies (stored data stays untouched)
        tasks, t_notes = scan_tasks(get_project_tasks(store))
        members, m_notes = scan_members(get_team_members(store))
        comments, c_notes = scan_comments(get_project_comments(store))
        deadline = get_project_deadline(store)
        scan_notes = t_notes + m_notes + c_notes
        alerts.extend(scan_notes)
        rec("Load data + safety scan", "guard", "warning" if scan_notes else "ok",
            f"{len(tasks)} task(s), {len(members)} member(s), {len(comments)} comment(s)"
            + (f"; {len(scan_notes)} hidden instruction(s) removed" if scan_notes else ""))

        effective_budget = (
            update.budget_used_percent if update.budget_used_percent is not None else get_project_budget(store)
        )

        # Low-signal shortcut: ONLY when normal route + zero state changes + greeting without specific questions
        has_new_state = bool(
            update.tasks
            or update.blockers
            or update.risks
            or update.budget_used_percent is not None
            or update.mentioned_deadline is not None
        )
        is_greeting = bool(GREETING_PATTERN.search(clean_msg.text.strip())) and not bool(
            QUESTION_WORDS.search(clean_msg.text)
        )
        is_low_signal = (route == Route.NORMAL) and (not has_new_state) and is_greeting

        if is_low_signal:
            plan = OrchestrationPlan(workers=["tasks", "people", "deadline"], focus="Current project health overview")
            rec("Orchestrator", "code", "ok", "Fast-path: selected deterministic metrics for greeting/status check",
                plan.model_dump_json(indent=1))

            results, (worker_reports, worker_gaps) = await asyncio.gather(
                run_all_checks(
                    tasks, members, deadline, today,
                    workers=tuple(plan.workers), budget_used_percent=effective_budget,
                ),
                run_worker_agents(list(plan.workers), ctx, runner=rules_worker_runner),
            )
            results.data_gaps.extend(extra_gaps + worker_gaps)
            worker_notes = [f"{k}: " + " | ".join(r.findings + r.concerns) for k, r in worker_reports.items()]
            bits = _result_bits(results)
            rec("Parallel workers + checks", "code", "warning" if results.data_gaps else "ok",
                "; ".join(bits) + (f". GAPS: {'; '.join(results.data_gaps)}" if results.data_gaps else ""),
                results.model_dump_json(indent=1) + "\n\nworker reports:\n"
                + "\n".join(f"{k}: {r.model_dump_json()}" for k, r in worker_reports.items()))

            if results.deadline and results.deadline.critical:
                route = ctx.route = Route.CRITICAL_PATH
                alerts.append(f"Deadline at risk: {results.deadline.reason}")
                rec("Deadline rules", "code", "warning", results.deadline.reason)
            else:
                rec("Deadline rules", "code", "ok" if results.deadline else "skipped",
                    results.deadline.reason if results.deadline else "Deadline check skipped")

            risks = _rank(deterministic_risks(update, results))
            gaps = list(results.data_gaps)
            rec("Risk analysis", "code", "ok",
                f"Fast-path: {len(risks)} deterministic risk(s)"
                + (f"; top: {risks[0].title} ({risks[0].severity.value})" if risks else ""),
                "[" + ",\n".join(r.model_dump_json() for r in risks) + "]")

            return await _synthesize_and_execute(
                update, route, plan, tasks, results, worker_reports, worker_notes,
                risks, gaps, set(), allow_actions=False, evaluator_llm=None,
            )

        # Normal full path
        plan, note = await plan_work(update, clean_msg.text,
                                     force_deadline=(route == Route.CRITICAL_PATH or bool(update.blockers)), llm=llm)
        rec("Orchestrator", "llm", "warning" if note else "ok",
            f"workers to run: {', '.join(plan.workers)}. Focus: {plan.focus}" + (f" ({note})" if note else ""),
            plan.model_dump_json(indent=1))

        results, (worker_reports, worker_gaps) = await asyncio.gather(
            run_all_checks(
                tasks, members, deadline, today,
                workers=tuple(plan.workers), budget_used_percent=effective_budget,
            ),
            run_worker_agents(list(plan.workers), ctx, runner=worker_runner),
        )
        results.data_gaps.extend(extra_gaps + worker_gaps)
        bits = _result_bits(results)
        worker_notes = [f"{k}: " + " | ".join(r.findings + r.concerns) for k, r in worker_reports.items()]
        rec("Parallel workers + checks", "llm" if worker_runner else "code", "warning" if results.data_gaps else "ok",
            "; ".join(bits) + (f". GAPS: {'; '.join(results.data_gaps)}" if results.data_gaps else ""),
            results.model_dump_json(indent=1) + "\n\nworker reports:\n"
            + "\n".join(f"{k}: {r.model_dump_json()}" for k, r in worker_reports.items()))

        if results.deadline and results.deadline.critical:
            route = ctx.route = Route.CRITICAL_PATH
            alerts.append(f"Deadline at risk: {results.deadline.reason}")
            rec("Deadline rules", "code", "warning", results.deadline.reason)
        else:
            rec("Deadline rules", "code", "ok" if results.deadline else "skipped",
                results.deadline.reason if results.deadline else "Deadline check was not needed for this request")

        risk_out = await analyze_risks_detailed(update, results, llm=llm, notes=worker_notes)
        risks, gaps, model_titles = risk_out.risks, risk_out.gaps, risk_out.model_titles
        rec("Risk analysis", "llm", "ok",
            f"{len(risks)} risk(s)" + (f"; top: {risks[0].title} ({risks[0].severity.value})" if risks else ""),
            "[" + ",\n".join(r.model_dump_json() for r in risks) + "]")

        return await _synthesize_and_execute(
            update, route, plan, tasks, results, worker_reports, worker_notes,
            risks, gaps, model_titles, allow_actions=True, evaluator_llm=llm,
        )
    except ConfigError as exc:
        rec("Configuration", "code", "error", redact(exc))
        return finish("error", f"Configuration error: {redact(exc)}")
    except ProviderCallError as exc:
        safe_msg = redact(exc.user_message)
        rec("Provider", "llm", "error", safe_msg)
        return finish("error", safe_msg)
    except Exception as exc:  # last resort: never crash the app, never invent a result
        rec("Pipeline", "code", "error", f"{type(exc).__name__}: {redact(exc)}")
        return finish("error", f"Something went wrong ({type(exc).__name__}). Nothing was executed after this point.")
