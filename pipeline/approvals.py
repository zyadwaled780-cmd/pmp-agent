"""Manager approval queue. Nothing risky runs until a person approves it here.

Decision rules:
* Unknown / empty ids raise ``UnknownApprovalError`` (a domain error) from approve()/reject(), and are
  reported as ``DecisionResult(ok=False)`` by resolve(), which never raises for bad input.
* A decision is applied AT MOST ONCE. The first decision wins; any later approve/reject of the same item
  (repeat or competing) is a no-op that returns the existing outcome.
* Approval executes against the CURRENT project state (the executor validates again), so a stale approval
  fails safely instead of acting on old assumptions.
* All timestamps are timezone-aware UTC instants.
"""
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

from domain.models import ProposedAction
from pipeline.adapters import Route
from tools.execution_tools import ExecutionResult, ProjectStore, execute_action


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ApprovalError(Exception):
    """Base class for approval-queue domain errors."""


class UnknownApprovalError(ApprovalError, LookupError):
    """No pending item has this id."""


class PendingApproval(BaseModel):
    id: str
    action: ProposedAction
    route: Route
    reason: str
    created_at: datetime = Field(default_factory=_now)
    decided_at: Optional[datetime] = None
    status: Literal["pending", "executing", "approved", "rejected", "failed", "interrupted"] = "pending"
    result: Optional[ExecutionResult] = None
    decision_note: str = ""


class DecisionResult(BaseModel):
    ok: bool                    # the id was valid
    applied: bool               # THIS call made the decision (False for repeats / competing decisions)
    message: str
    item: Optional[PendingApproval] = None


class ApprovalQueue:
    def __init__(self):
        self._items: dict[str, PendingApproval] = {}
        self._lock = threading.RLock()

    def __deepcopy__(self, memo):
        cloned = ApprovalQueue()
        cloned.restore_from(self.snapshot())
        return cloned

    def add(self, action: ProposedAction, route: Route, reason: str) -> str:
        with self._lock:
            pid = uuid.uuid4().hex[:6]
            while pid in self._items:
                pid = uuid.uuid4().hex[:6]
            self._items[pid] = PendingApproval(id=pid, action=action, route=route, reason=reason)
            return pid

    def get(self, pid: str) -> Optional[PendingApproval]:
        with self._lock:
            return self._items.get((pid or "").strip())

    def pending(self) -> list[PendingApproval]:
        with self._lock:
            return [i for i in self._items.values() if i.status == "pending"]

    def all(self) -> list[PendingApproval]:
        with self._lock:
            return list(self._items.values())

    def snapshot(self) -> list[PendingApproval]:
        """Return deep copies of all items in insertion order."""
        with self._lock:
            return [item.model_copy(deep=True) for item in self._items.values()]

    def restore_from(self, items: list[PendingApproval]) -> None:
        """Replace queue contents cleanly from persisted items."""
        with self._lock:
            restored: dict[str, PendingApproval] = {}
            for item in items:
                copy = item.model_copy(deep=True)
                if copy.status == "executing":
                    copy.status = "interrupted"
                    copy.decision_note = (
                        copy.decision_note
                        or "Interrupted mid-execution before durable commit; verify state manually."
                    )
                    copy.decided_at = copy.decided_at or _now()
                restored[copy.id] = copy
            self._items = restored

    def _require(self, pid: str) -> PendingApproval:
        item = self._items.get((pid or "").strip())
        if item is None:
            raise UnknownApprovalError(f"no approval item with id '{str(pid)[:20]}'")
        return item

    def approve(self, pid: str, store: ProjectStore, note: str = "") -> PendingApproval:
        with self._lock:
            item = self._require(pid)
            if item.status != "pending":
                return item                               # repeat / competing decision: first one wins
            item.result = execute_action(item.action, store, authorized=True)
            item.status = "approved" if item.result.status == "confirmed" else "failed"
            item.decision_note = note
            item.decided_at = _now()
            return item

    def reject(self, pid: str, note: str = "") -> PendingApproval:
        with self._lock:
            item = self._require(pid)
            if item.status == "pending":
                item.status = "rejected"
                item.decision_note = note or "Rejected by manager"
                item.decided_at = _now()
            return item

    def resolve(self, pid: str, approve: bool, store: ProjectStore, note: str = "") -> DecisionResult:
        """Never raises for bad input: used by the UI."""
        with self._lock:
            item = self._items.get((pid or "").strip())
            if item is None:
                return DecisionResult(ok=False, applied=False, message="Unknown approval id.")
            if item.status != "pending":
                return DecisionResult(ok=True, applied=False, item=item,
                                      message=f"Already decided ({item.status}); nothing was changed.")
            item = self.approve(pid, store, note) if approve else self.reject(pid, note)
            detail = item.result.detail if item.result else item.decision_note
            return DecisionResult(ok=True, applied=True, item=item, message=f"{item.status}: {detail}")
