"""The ONLY file that talks to the partner's core/ (guardrails, router, action gate).

His classes work on his models (core.models); the agent layer uses domain.models. This file
translates between them and adds three small extras his core does not have (marked EXTRA):
bulk-delete requests, extra promise wording, and scanning project data with his injection patterns.
"""
import re
import uuid
from typing import Optional

from pydantic import BaseModel

from core import (
    ActionGate,
    ActionPolicyVerdict,
    ActionType,
    DataSafetyGuardrail,
    DeterministicRouter,
    InputGuardrail,
    OutputGuardrail,
    RouteDecision,
    RouteType,
    UpdateSource,
)
from core import ExtractedUpdate as CoreUpdate
from core import InputMessage as CoreMessage
from core import ProposedAction as CoreAction
from core import TaskItem as CoreTask
from domain.models import ExecutiveReport, ExtractedUpdate, ProposedAction, TaskItem
from project_agents.claims import has_completion_claim, has_delivery_promise

Route = RouteType
Verdict = ActionPolicyVerdict
REDACTED = "[REDACTED_SUSPICIOUS_PAYLOAD]"

# EXTRA: wider "override the rules" wording than his patterns (e.g. "disregard all the rules")
OVERRIDE = [re.compile(p, re.I) for p in [
    r"\b(disregard|forget|bypass|override|skip)\b.{0,20}\b(rules|instructions|guidelines|policy|policies|restrictions)\b",
    r"(انسى|تجاهل|تخطى|اتجاهل)\s+(كل\s+)?(القواعد|التعليمات|القوانين)",
]]

# EXTRA: bulk destructive requests that are not a "prompt injection" but must be refused
DESTRUCTIVE = [re.compile(p, re.I) for p in [
    r"\b(delete|remove|drop|erase)\b.{0,25}\b(all|every|whole|entire|backlog|sprint|project)\b",
    r"\bwipe\b",
    r"(احذف|امسح|شيل) (كل|جميع|المشروع|الباكلوج)",
]]
# EXTRA: promise wording beyond his patterns
PROMISE = re.compile(r"\bwe (will|'ll) (deliver|ship|finish|complete)\b|\bguarantee[sd]?\b|\bpromise[sd]?\b|هنسلم يوم|نضمن", re.I)


class GuardrailResult(BaseModel):
    ok: bool
    category: Optional[str] = None   # injection | destructive | empty | output
    reason: str = ""


def _msg(text: str, source: str = "unknown", sender: str = "unknown") -> CoreMessage:
    try:
        src = UpdateSource(source.upper())
    except ValueError:
        src = UpdateSource.UNKNOWN
    return CoreMessage(message_id=uuid.uuid4().hex[:8], source=src, sender_id=sender, text=text)


def input_guardrail(text: str, source: str = "unknown", sender: str = "unknown") -> GuardrailResult:
    if not text or not text.strip():
        return GuardrailResult(ok=False, category="empty", reason="Empty message.")
    a = InputGuardrail.scan_message(_msg(text, source, sender))
    if not a.is_safe:
        return GuardrailResult(ok=False, category="injection", reason=a.reason or "Prompt injection detected.")
    if any(p.search(text) for p in OVERRIDE):
        return GuardrailResult(ok=False, category="injection", reason="Message tries to override the rules.")
    if any(p.search(text) for p in DESTRUCTIVE):
        return GuardrailResult(ok=False, category="destructive", reason="Message asks for a destructive bulk action.")
    return GuardrailResult(ok=True)


def scan_text(text: Optional[str]) -> tuple[Optional[str], bool]:
    """Data safety scan for text that came from the project data (titles, blockers, comments)."""
    if not text:
        return text, False
    cleaned, violated, _ = DataSafetyGuardrail.sanitize_external_data(text)
    # EXTRA: data must not contain the instruction-override phrases that are blocked in user input
    if not InputGuardrail.scan_message(_msg(cleaned)).is_safe or any(p.search(cleaned) for p in OVERRIDE):
        return REDACTED, True
    return cleaned, violated


def scan_tasks(tasks: list[TaskItem]) -> tuple[list[TaskItem], list[str]]:
    """Model-facing COPIES of the tasks: title, blocker and owner are scanned. The stored tasks are untouched."""
    clean, notes = [], []
    for t in tasks:
        t = t.model_copy(deep=True)
        clean_title, f1 = scan_text(t.title)
        t.title = clean_title or ""
        t.blocker, f2 = scan_text(t.blocker)
        t.owner, f3 = scan_text(t.owner)
        if f1 or f2 or f3:
            notes.append(f"Hidden instruction removed from task {t.id or '?'}")
        clean.append(t)
    return clean, notes


def scan_members(members: list) -> tuple[list, list[str]]:
    """Model-facing COPIES of the team list; names carrying instructions are redacted (stored data untouched)."""
    clean, notes = [], []
    for i, m in enumerate(members, 1):
        m = m.model_copy(deep=True)
        m.name, flagged = scan_text(m.name)
        if flagged:
            notes.append(f"Hidden instruction removed from team member #{i}")
        clean.append(m)
    return clean, notes


def scan_comments(comments: list[dict]) -> tuple[list[dict], list[str]]:
    """Model-facing COPIES of stored comments; instructions inside a comment are redacted, never obeyed."""
    clean, notes = [], []
    for i, c in enumerate(comments, 1):
        text, flagged = scan_text(str(c.get("text", "")))
        clean.append({**c, "text": text})
        if flagged:
            notes.append(f"Hidden instruction removed from comment #{i}")
    return clean, notes


_SENDER_MAX = 80


def clean_sender(name: str) -> tuple[str, bool]:
    """The sender label ends up in prompts: flatten it, cap its length and redact instructions."""
    flat = " ".join(str(name or "unknown").split())[:_SENDER_MAX] or "unknown"
    text, flagged = scan_text(flat)
    return text or "unknown", flagged


_STATUS = {"todo": "CREATED", "in_progress": "IN_PROGRESS", "blocked": "BLOCKED", "done": "COMPLETED"}


def route_update(update: ExtractedUpdate, original_text: str = "") -> RouteDecision:
    """Translate the update into his model and let his deterministic router decide."""
    tasks = [
        CoreTask(task_id=t.id or f"new-{i}", title=t.title, owner=t.owner,
                 due_date=str(t.due_date) if t.due_date else None, status=_STATUS[t.status],
                 is_blocker=(t.status == "blocked" or bool(t.blocker)), delay_days=t.delay_days or 0)
        for i, t in enumerate(update.tasks)
    ]
    if update.blockers and not any(t.is_blocker for t in tasks):
        # a blocker reported without a matching task still counts as a blocker
        tasks.append(CoreTask(task_id="reported-blocker", title=update.blockers[0][:80],
                              status="BLOCKED", is_blocker=True))
    core = CoreUpdate(update_id=uuid.uuid4().hex[:8], source=UpdateSource(update.source.value.upper()),
                      tasks=tasks, raw_entities={"raw_text": original_text})
    return DeterministicRouter.evaluate_route(core)


def _to_core_action(a: ProposedAction) -> CoreAction:
    if a.action == "create_project_comment":
        kind, payload = ActionType.ADD_COMMENT, {"comment": a.comment}
    elif a.action == "create_task":
        kind, payload = ActionType.CREATE_TASK, {"title": a.title, "owner": a.owner}
    elif a.action == "assign_task":
        kind, payload = ActionType.REASSIGN_TASK, {"owner": a.owner}
    elif a.action == "update_task":
        kind = ActionType.UPDATE_DUE_DATE if a.due_date else ActionType.CUSTOM_API
        payload = {"status": a.status, "due_date": str(a.due_date) if a.due_date else None}
    else:  # delete_task and anything unknown
        kind, payload = ActionType.DELETE_TASK, {}
    return CoreAction(action_id=uuid.uuid4().hex[:8], action_type=kind, target=a.task_id or "project",
                      payload=payload, justification=a.reason)


def gate_action(action: ProposedAction, route: RouteType) -> tuple[ActionPolicyVerdict, str]:
    return ActionGate.evaluate_action(_to_core_action(action), route=route)


def check_output_text(text: str) -> GuardrailResult:
    ok, issues = OutputGuardrail.validate_output(text, is_system_verified=False)
    if ok and (PROMISE.search(text) or has_delivery_promise(text)):
        ok, issues = False, ["Report promises a delivery date or outcome."]
    if ok and has_completion_claim(text):
        ok, issues = False, ["Report claims completion that no data supports."]
    return GuardrailResult(ok=ok, category=None if ok else "output", reason="; ".join(issues))


def output_guardrail(report: ExecutiveReport) -> GuardrailResult:
    return check_output_text("\n".join([report.summary] + [a.comment or "" for a in report.proposed_actions]))
