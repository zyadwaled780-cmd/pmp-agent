"""Step 12 - reject any sentence or action that the data does not support (bounded, targeted revisions).

Who owns what (a failure is routed to the component that can actually fix it):
* ``writer``   - summary and proposed_actions            -> Status Writer revises (with only ITS feedback)
* ``risk``     - model-generated risks (not deterministic) -> Risk Analyzer re-checks / removes ungrounded ones
* ``pipeline`` - anything no component can change (e.g. a dispute about a deterministic risk) -> stop at once,
                 escalate to the manager with the reason. Retrying cannot help.

Pass/fail is NOT left to the LLM alone: Python grounding checks always run and the report passes only if both
agree. The loop is bounded (``max_iterations``), stops early on repeated no-progress failures and never turns
an exhausted loop into a "successful" report: the outcome is ``approved=False`` with an actionable reason.
"""
import re
from typing import Literal, Optional

from pydantic import BaseModel

from checks import CheckResults
from domain.models import ExecutiveReport, ExtractedUpdate, RiskItem, TaskItem
from project_agents.claims import has_completion_claim, has_delivery_promise, normalize_text
from project_agents.llm import LLMCall, call_llm
from project_agents.risk_analyzer import _content_tokens, fact_corpus, revise_model_risks, ungrounded_reason
from project_agents.status_writer import write_report

MAX_ITERATIONS = 3
NO_PROGRESS_REPEATS = 2          # the identical failure after this many consecutive revisions = no progress
_SAME = object()


class EvalResult(BaseModel):
    passed: bool
    issues: list[str]
    feedback: str


class Finding(BaseModel):
    owner: Literal["writer", "risk", "pipeline"]
    message: str
    titles: list[str] = []          # model-risk titles this finding is about (owner == "risk")


INSTRUCTIONS = """You are a strict reviewer of a project status report. Input: the facts (updates + checks) and the report.
Fail the report if ANY sentence or proposed action is not supported by the facts, if it claims something is finished
without status "done", if it promises a delivery date, or if it contradicts the data.
Recommendations ("should", "consider", "ask the supplier") are allowed; unsupported statements of fact are not.
Do not fail for valid arithmetic or minor stylistic choices.
Return passed=true only if everything is supported. issues: one short line per problem. feedback: what to change.
Start an issue with "RISK:" ONLY when the problem is in the report's risks list (name the risk title), never for the
summary or actions."""

_TASK_LINE = re.compile(r"^(?P<id>[^:]+): (?P<title>.*) \((?P<status>\w+)\)$")


# --------------------------------------------------------------------------- state helpers
def _state(results: CheckResults, tasks: Optional[list[TaskItem]]) -> dict[str, TaskItem]:
    by_id = {t.id: t for t in (tasks or []) if t.id}
    if not by_id and results.tasks:
        for t in [*results.tasks.overdue, *results.tasks.blocked, *results.tasks.due_after_deadline]:
            if t.id:
                by_id.setdefault(t.id, t)
    return by_id


def _titles(results: CheckResults, state: dict[str, TaskItem]) -> dict[str, str]:
    titles = {i: t.title for i, t in state.items()}
    for line in (results.tasks.task_catalog if results.tasks else []):
        m = _TASK_LINE.match(line)
        if m:
            titles.setdefault(m["id"], m["title"])
    return titles


def _norm(s: Optional[str]) -> str:
    return " ".join(normalize_text(s or "").split())


def _matches(update: ExtractedUpdate, task_id: Optional[str], title: Optional[str]):
    nt = _norm(title)
    return [t for t in update.tasks
            if (t.id and t.id == task_id) or (nt and _norm(t.title) and (_norm(t.title) in nt or nt in _norm(t.title)))]


def _mentions(update: ExtractedUpdate, task_id: Optional[str], title: Optional[str]) -> bool:
    nt = _norm(title)
    texts = [_norm(b) for b in update.blockers] + [_norm(r) for r in update.risks]
    return any((nt and nt in t) or (task_id and _norm(task_id) in t) for t in texts)


# --------------------------------------------------------------------------- deterministic grounding
def grounding_findings(
    report: ExecutiveReport,
    update: ExtractedUpdate,
    results: CheckResults,
    tasks: Optional[list[TaskItem]] = None,
    notes: Optional[list[str]] = None,
) -> list[Finding]:
    out: list[Finding] = []

    def add(msg: str, owner: Literal["writer", "risk", "pipeline"] = "writer", titles: Optional[list[str]] = None):
        out.append(Finding(owner=owner, message=msg, titles=titles or []))

    known_ids = results.known_task_ids()
    people = results.known_people()
    unavailable = set(results.people.unavailable) if results.people else set()
    overloaded = set(results.people.overloaded) if results.people else set()
    state = _state(results, tasks)
    titles = _titles(results, state)
    done_in_update = {_norm(t.title) for t in update.tasks if t.status == "done"}
    done_ids_in_update = {t.id for t in update.tasks if t.status == "done" and t.id}

    # ---- summary (writer) -------------------------------------------------------------------------
    if not report.summary.strip():
        add("Summary is empty.")
    open_count = results.tasks.open_count if results.tasks else 0
    if has_completion_claim(report.summary) and open_count > 0:
        add(f"Summary claims completion but {open_count} task(s) are still open.")
    if has_delivery_promise(report.summary):
        add("Summary promises a delivery or outcome that no data supports.")

    # ---- proposed actions (writer), grounded per kind --------------------------------------------
    for i, a in enumerate(report.proposed_actions, 1):
        tag = f"Action {i} ({a.action})"
        if not a.reason.strip() or not a.evidence.strip():
            add(f"{tag} has no reason or evidence.")
        if a.task_id and known_ids and a.task_id not in known_ids and a.action not in ("update_task", "assign_task"):
            add(f"{tag} uses unknown task_id '{a.task_id}'.")
        cur = state.get(a.task_id) if a.task_id else None
        title = titles.get(a.task_id or "", None)

        if a.action in ("update_task", "assign_task"):
            if not a.task_id:
                add(f"{tag} has no task_id.")
            elif known_ids and a.task_id not in known_ids:
                add(f"{tag} uses unknown task_id '{a.task_id}'.")

        if a.action == "assign_task":
            if not a.owner:
                add(f"{tag} has no owner.")
            elif people and a.owner not in people:
                add(f"{tag} assigns to '{a.owner}' who is not in the team.")
            elif a.owner in unavailable:
                add(f"{tag} assigns to '{a.owner}' who is on leave.")
            elif cur is not None:
                reported = any(t.owner == a.owner for t in _matches(update, a.task_id, title))
                reason_in_checks = cur.owner is None or cur.owner in overloaded or cur.owner in unavailable
                if cur.owner == a.owner:
                    add(f"{tag} assigns the task to its current owner '{a.owner}'.")
                elif not (reported or reason_in_checks):
                    add(f"{tag} reassigns '{a.task_id}' without evidence: the update does not ask for it and its "
                        f"current owner is neither overloaded nor unavailable.")

        elif a.action == "update_task" and a.task_id and (not known_ids or a.task_id in known_ids):
            matched = _matches(update, a.task_id, title)
            if a.status == "done":
                if a.task_id not in done_ids_in_update and _norm(a.title) not in done_in_update \
                        and not any(t.status == "done" for t in matched):
                    add(f"{tag} marks a task done but no update says it is finished.")
            elif a.status is not None:
                said = any(t.status == a.status for t in matched)
                if a.status == "blocked":
                    said = said or any(t.blocker for t in matched) or _mentions(update, a.task_id, title)
                if not said:
                    add(f"{tag} sets status '{a.status}' on '{a.task_id}' but the update does not report it.")
            if a.due_date is not None:
                said = any(t.due_date == a.due_date for t in matched) or update.mentioned_deadline == a.due_date
                if not said and cur is not None and cur.due_date is not None:
                    said = any(t.delay_days and a.due_date == cur.due_date.fromordinal(cur.due_date.toordinal() + t.delay_days)
                               for t in matched)
                if not said:
                    add(f"{tag} changes the due date of '{a.task_id}' to {a.due_date} but the update does not report it.")

        elif a.action == "create_task":
            title_text = a.title or ""
            if not title_text.strip():
                add(f"{tag} has no title.")
            else:
                nt = _norm(title_text)
                if any(_norm(v) == nt for v in titles.values()):
                    add(f"{tag} duplicates an existing task titled '{title_text}'.")
                reported = bool(_matches(update, None, title_text))
                if not reported:
                    wanted = set(_content_tokens(title_text))
                    heard = set(_content_tokens(" ".join([update.summary, *update.blockers, *update.risks])))
                    reported = bool(wanted) and len(wanted & heard) / len(wanted) >= 0.5
                if not reported:
                    add(f"{tag} creates '{title_text}' but nothing in the update asks for or implies it.")
            if a.owner and people and a.owner not in people:
                add(f"{tag} assigns the new task to '{a.owner}' who is not in the team.")
            elif a.owner and a.owner in unavailable:
                add(f"{tag} assigns the new task to '{a.owner}' who is on leave.")

        elif a.action == "create_project_comment":
            comment_text = a.comment or ""
            if not comment_text.strip():
                add(f"{tag} has no comment text.")
            else:
                if has_completion_claim(comment_text) and open_count > 0:
                    add(f"{tag} claims completion that no data supports.")
                if has_delivery_promise(comment_text):
                    add(f"{tag} promises a delivery or outcome that no data supports.")
            if a.task_id and known_ids and a.task_id not in known_ids:
                pass   # already reported as unknown task above

    # ---- risks: model-generated ones must be grounded (risk analyzer owns them) -------------------
    corpus = fact_corpus(update, results, notes)
    by_title = {r.title: r for r in report.risks}
    for t in report.model_risk_titles:
        r = by_title.get(t)
        why = ungrounded_reason(r, corpus, known_ids) if r else None
        if why:
            add(f"RISK: '{t[:60]}' is not grounded in the facts ({why}).", "risk", [t])
    return out


def grounding_issues(report: ExecutiveReport, update: ExtractedUpdate, results: CheckResults,
                     tasks: Optional[list[TaskItem]] = None) -> list[str]:
    """Plain messages (kept for callers that only need the text)."""
    return [f.message for f in grounding_findings(report, update, results, tasks)]


# --------------------------------------------------------------------------- evaluation
async def evaluate_report_detailed(
    report: ExecutiveReport,
    update: ExtractedUpdate,
    results: CheckResults,
    llm: Optional[LLMCall] = call_llm,
    tasks: Optional[list[TaskItem]] = None,
    notes: Optional[list[str]] = None,
) -> tuple[EvalResult, list[Finding]]:
    findings = grounding_findings(report, update, results, tasks, notes)
    llm_passed, note = True, ""
    if not findings and llm is not None:
        prompt = (f"facts: update={update.model_dump_json(exclude_none=True)} checks={results.model_dump_json(exclude_none=True)}\n"
                  f"report: {report.model_dump_json(exclude_none=True)}\n"
                  f"model_generated_risks: {report.model_risk_titles}")
        try:
            ev = await llm("Evaluator", INSTRUCTIONS, prompt, EvalResult)
            llm_passed = ev.passed
            for issue in ev.issues:
                findings.append(_classify_llm_issue(issue, report))
            if not ev.passed and not ev.issues:
                findings.append(Finding(owner="writer", message="The reviewer rejected the report without giving details."))
        except Exception as exc:
            note = f" (LLM evaluator unavailable: {type(exc).__name__}; only rule checks ran)"
    messages = [f.message for f in findings]
    passed = llm_passed and not findings
    feedback = "OK" + note if passed else "Fix: " + "; ".join(messages) + note
    return EvalResult(passed=passed, issues=messages, feedback=feedback), findings


def _classify_llm_issue(issue: str, report: ExecutiveReport) -> Finding:
    if not issue.strip().upper().startswith("RISK:"):
        return Finding(owner="writer", message=issue)
    low = issue.lower()
    titles = [t for t in report.model_risk_titles if t.lower() in low]
    if titles:
        return Finding(owner="risk", message=issue, titles=titles)
    return Finding(owner="pipeline", message=(
        issue + " [not attributable to a model-generated risk: deterministic risks come from the checks and "
                "cannot be changed by a retry]"))


async def evaluate_report(
    report: ExecutiveReport,
    update: ExtractedUpdate,
    results: CheckResults,
    llm: Optional[LLMCall] = call_llm,
) -> EvalResult:
    return (await evaluate_report_detailed(report, update, results, llm))[0]


# --------------------------------------------------------------------------- the bounded loop
class LoopOutcome(BaseModel):
    report: ExecutiveReport
    evaluation: EvalResult
    iterations: int
    approved: bool
    escalate_to_human: bool
    termination_reason: str = "approved"      # approved | max_iterations | no_progress | unfixable
    unresolved: list[str] = []


def _escalate(report: ExecutiveReport, ev: EvalResult, i: int, reason: str, issues: list[str]) -> LoopOutcome:
    note = {"max_iterations": "the review limit was reached",
            "no_progress": "the same problems came back after revisions",
            "unfixable": "the problems cannot be fixed by revising"}[reason]
    flagged = report.model_copy(update={
        "review_status": "needs_review",
        "data_gaps": [*report.data_gaps, f"NOT APPROVED by the evaluator ({note}); the manager must review: " + "; ".join(issues)],
    })
    return LoopOutcome(report=flagged, evaluation=ev, iterations=i, approved=False, escalate_to_human=True,
                       termination_reason=reason, unresolved=issues)


async def write_and_evaluate(
    update: ExtractedUpdate,
    risks: list[RiskItem],
    results: CheckResults,
    data_gaps: list[str],
    llm: LLMCall = call_llm,
    max_iterations: int = MAX_ITERATIONS,
    language: str = "English",
    notes: Optional[list[str]] = None,
    *,
    model_risk_titles: Optional[set[str]] = None,
    evaluator_llm=_SAME,
    tasks: Optional[list[TaskItem]] = None,
    allow_actions: bool = True,
    no_progress_repeats: int = NO_PROGRESS_REPEATS,
) -> LoopOutcome:
    """Writer <-> Evaluator loop with routed revisions. Pass ``evaluator_llm=None`` to run only the Python checks."""
    eval_llm = llm if evaluator_llm is _SAME else evaluator_llm
    current_risks, titles, gaps = list(risks), set(model_risk_titles or ()), list(data_gaps)
    report: Optional[ExecutiveReport] = None
    writer_feedback: Optional[str] = None
    need_write, prev_sig, repeats = True, None, 0

    for i in range(1, max_iterations + 1):
        if need_write or report is None:
            report = await write_report(update, current_risks, results, gaps, feedback=writer_feedback, previous=report,
                                        llm=llm, language=language, notes=notes, allow_actions=allow_actions,
                                        model_risk_titles=titles)
        else:       # only risks changed: the writer has nothing to fix, re-check the same report with the new risks
            report = report.model_copy(update={"risks": current_risks, "data_gaps": list(gaps),
                                               "model_risk_titles": sorted(titles)})
        ev, findings = await evaluate_report_detailed(report, update, results, llm=eval_llm, tasks=tasks, notes=notes)
        if ev.passed:
            return LoopOutcome(report=report, evaluation=ev, iterations=i, approved=True, escalate_to_human=False)

        messages = [f.message for f in findings]
        if any(f.owner == "pipeline" for f in findings):
            return _escalate(report, ev, i, "unfixable", messages)
        sig = frozenset(" ".join(m.lower().split()) for m in messages)
        repeats = repeats + 1 if sig == prev_sig else 0
        prev_sig = sig
        if repeats >= no_progress_repeats:
            return _escalate(report, ev, i, "no_progress", messages)
        if i == max_iterations:
            return _escalate(report, ev, i, "max_iterations", messages)

        risk_f = [f for f in findings if f.owner == "risk"]
        writer_f = [f for f in findings if f.owner == "writer"]
        if risk_f:
            out = await revise_model_risks(update, results, current_risks, titles,
                                           {t for f in risk_f for t in f.titles},
                                           "; ".join(f.message for f in risk_f), llm, notes)
            current_risks, titles = out.risks, out.model_titles
            gaps = [*gaps, *[g for g in out.gaps if g not in gaps]]
        need_write = bool(writer_f)
        if writer_f:
            writer_feedback = "; ".join(f.message for f in writer_f)
    raise AssertionError("unreachable")  # pragma: no cover
