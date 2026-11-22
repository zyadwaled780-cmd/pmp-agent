"""Step 15 - actually perform the actions that the action gate allowed.

Rules:
- Runs only when `authorized=True` (the gate said ALLOWED, or the manager approved).
  This is a second lock, not a replacement for the gate.
- delete_task and unknown actions are always refused.
- Every action is VALIDATED AGAIN against the current store state, under the store lock, right before it
  is applied (tools.validation.validate_action). An invalid action changes nothing.
- Retries only transient failures (TransientToolError, TimeoutError, OSError), twice.
  Mistakes such as "task not found" fail at once, retrying cannot fix them.
- Reports "confirmed" only after reading the change back from the store.

ProjectStore is an in-memory stand-in; swap it for SQLite/JSON later with the same methods.
Each session owns its own ProjectStore (see pipeline/session.py); compound operations hold ``store.lock``.
"""
import copy
import re
import threading
import time
from datetime import date, datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel

from checks.budget_checker import validate_budget_percent
from checks.team_checker import TeamMember
from domain.models import ProposedAction, TaskItem, TaskStatus
from tools.validation import validate_action

_ID_NUMBER = re.compile(r"^T(\d+)$")


class TransientToolError(Exception):
    """Temporary failure (timeout, lock, network). Safe to retry."""


class ProjectStore:
    def __init__(self, tasks: Optional[list[TaskItem]] = None,
                 members: Optional[list[TeamMember]] = None, deadline: Optional[date] = None):
        self.lock = threading.RLock()
        self.tasks: dict[str, TaskItem] = {t.id: t for t in (tasks or []) if t.id}
        self.members: list[TeamMember] = list(members or [])
        self.deadline: Optional[date] = deadline
        self.comments: list[dict] = []
        # budget state (see set_budget): latest accountant-reported % of budget used
        self.budget_used_percent: Optional[float] = None
        self.budget_history: list[dict] = []
        self._counter = max(len(self.tasks), self._max_numeric_id())     # high-water mark: ids are never reused

    def __deepcopy__(self, memo):
        with self.lock:
            cloned = ProjectStore(
                tasks=[t.model_copy(deep=True) for t in self.tasks.values()],
                members=[m.model_copy(deep=True) for m in self.members],
                deadline=copy.deepcopy(self.deadline, memo),
            )
            cloned.comments = copy.deepcopy(self.comments, memo)
            cloned.budget_used_percent = self.budget_used_percent
            cloned.budget_history = copy.deepcopy(self.budget_history, memo)
            cloned._counter = self._counter
            return cloned

    def _max_numeric_id(self) -> int:
        nums = [int(m.group(1)) for k in self.tasks if (m := _ID_NUMBER.match(k))]
        return max(nums, default=0)

    def new_id(self) -> str:
        """Collision-free even with non-contiguous ids (T1, T3) and after deletions; thread safe."""
        with self.lock:
            n = max(self._counter, self._max_numeric_id()) + 1
            while f"T{n}" in self.tasks:
                n += 1
            self._counter = n
            return f"T{n}"

    def set_budget(self, percent: float, source: str = "accountant") -> float:
        """Latest valid report replaces the stored value; every change is kept in ``budget_history``."""
        pct = validate_budget_percent(percent)
        with self.lock:
            self.budget_used_percent = pct
            self.budget_history.append({"percent": pct, "source": source,
                                        "at": datetime.now(timezone.utc).isoformat()})
        return pct


class ExecutionResult(BaseModel):
    action: str
    status: Literal["confirmed", "failed", "refused"]
    detail: str
    attempts: int = 0
    task_id: Optional[str] = None


# ---- the four write tools -------------------------------------------------

def create_task(store: ProjectStore, title: str, owner: Optional[str] = None,
                due_date: Optional[date] = None) -> str:
    if not title.strip():
        raise ValueError("title is required")
    tid = store.new_id()
    store.tasks[tid] = TaskItem(id=tid, title=title.strip(), owner=owner, status="todo", due_date=due_date)
    return tid


def update_task(store: ProjectStore, task_id: str, status: Optional[TaskStatus] = None,
                due_date: Optional[date] = None) -> str:
    task = store.tasks[task_id]  # KeyError if missing -> permanent failure
    if status is not None:
        task.status = status
    if due_date is not None:
        task.due_date = due_date
    return task_id


def assign_task(store: ProjectStore, task_id: str, owner: str) -> str:
    if not owner.strip():
        raise ValueError("owner is required")
    store.tasks[task_id].owner = owner
    return task_id


def create_project_comment(store: ProjectStore, comment: str, task_id: Optional[str] = None) -> str:
    if not comment.strip():
        raise ValueError("comment is required")
    if task_id is not None and task_id not in store.tasks:
        raise KeyError(task_id)
    store.comments.append({"task_id": task_id, "text": comment.strip()})
    return task_id or ""


# ---- executor --------------------------------------------------------------

def _run_once(a: ProposedAction, store: ProjectStore) -> tuple[str, Optional[str]]:
    """Validate against the CURRENT state, do the action, read it back. Atomic under the store lock."""
    with store.lock:
        error = validate_action(a, store)
        if error:
            raise ValueError(f"invalid at execution time: {error}")      # permanent: nothing was changed
        if a.action == "create_task":
            tid = create_task(store, a.title or "", a.owner, a.due_date)
            if store.tasks.get(tid) is None:
                raise TransientToolError("create_task not visible after write")
            return f"Created task {tid}", tid
        if a.action == "update_task":
            tid = update_task(store, a.task_id or "", a.status, a.due_date)
            t = store.tasks[tid]
            if (a.status and t.status != a.status) or (a.due_date and t.due_date != a.due_date):
                raise TransientToolError("update_task not visible after write")
            return f"Updated task {tid}", tid
        if a.action == "assign_task":
            tid = assign_task(store, a.task_id or "", a.owner or "")
            if store.tasks[tid].owner != a.owner:
                raise TransientToolError("assign_task not visible after write")
            return f"Assigned task {tid} to {a.owner}", tid
        if a.action == "create_project_comment":
            before = len(store.comments)
            tid = create_project_comment(store, a.comment or "", a.task_id)
            if len(store.comments) != before + 1:
                raise TransientToolError("comment not visible after write")
            return "Comment added", tid or None
    raise ValueError(f"unsupported action {a.action}")


def execute_action(action: ProposedAction, store: ProjectStore, *, authorized: bool,
                   max_retries: int = 2, backoff_seconds: float = 0.5) -> ExecutionResult:
    name = action.action
    if name == "delete_task":
        return ExecutionResult(action=name, status="refused", detail="delete_task is forbidden")
    if name not in ("create_task", "update_task", "assign_task", "create_project_comment"):
        return ExecutionResult(action=name, status="refused", detail=f"unknown action '{name}'")
    if not authorized:
        return ExecutionResult(action=name, status="refused", detail="not authorized by the action gate or the manager")

    attempts = 0
    last = ""
    while attempts <= max_retries:
        attempts += 1
        try:
            detail, tid = _run_once(action, store)
            return ExecutionResult(action=name, status="confirmed", detail=detail, attempts=attempts, task_id=tid)
        except (TransientToolError, TimeoutError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            if attempts <= max_retries and backoff_seconds:
                time.sleep(backoff_seconds * attempts)
        except Exception as exc:  # permanent: bad id, missing field, invalid at execution time ...
            return ExecutionResult(action=name, status="failed", detail=f"{type(exc).__name__}: {exc}", attempts=attempts)
    return ExecutionResult(action=name, status="failed",
                           detail=f"failed after {attempts} attempts ({last}); no result was invented", attempts=attempts)
