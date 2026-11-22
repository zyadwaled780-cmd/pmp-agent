"""The ONE protected path for changing the project (used by the pipeline and by the write tools).

validate arguments -> action gate -> (queue for the manager | execute with retry) -> honest outcome.
Nothing here claims success unless the executor confirmed it. The executor validates AGAIN against the
current state right before it applies the change (see tools.execution_tools), so a queued approval can
never act on stale assumptions.

This is the authoritative write path: no LLM agent receives a tool that bypasses it.
"""
from typing import Literal, Optional

from pydantic import BaseModel

from domain.models import ProposedAction
from pipeline.adapters import Verdict, gate_action
from pipeline.context import PMPContext
from tools.execution_tools import ExecutionResult, ProjectStore, execute_action
from tools.validation import validate_action as _validate


class ActionOutcome(BaseModel):
    status: Literal["executed", "queued", "blocked", "failed"]
    detail: str
    approval_id: Optional[str] = None
    execution: Optional[ExecutionResult] = None


def validate_action(a: ProposedAction, store: ProjectStore) -> Optional[str]:
    """Kept for compatibility: the rules live in tools.validation. Returns an error text or None."""
    with store.lock:
        return _validate(a, store)


def apply_action(action: ProposedAction, ctx: PMPContext, force_review: bool = False) -> ActionOutcome:
    label = action.action + (f" {action.task_id}" if action.task_id else "")
    verdict, why = gate_action(action, ctx.route)
    if verdict == Verdict.FORBIDDEN:
        ctx.audit.append(f"{label}: forbidden")
        return ActionOutcome(status="blocked", detail=f"forbidden ({why})")
    error = validate_action(action, ctx.store)
    if error:
        ctx.audit.append(f"{label}: invalid ({error})")
        return ActionOutcome(status="blocked", detail=f"invalid arguments: {error}")
    if verdict == Verdict.NEEDS_APPROVAL or force_review:
        reason = why if verdict == Verdict.NEEDS_APPROVAL else "the report needs manager review"
        pid = ctx.queue.add(action, ctx.route, reason)
        ctx.audit.append(f"{label}: queued {pid}")
        return ActionOutcome(status="queued", detail=reason, approval_id=pid)
    result = execute_action(action, ctx.store, authorized=True)
    ctx.audit.append(f"{label}: {result.status}")
    ok = result.status == "confirmed"
    return ActionOutcome(status="executed" if ok else "failed", detail=result.detail, execution=result)
