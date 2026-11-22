"""Per-session isolated state lifecycle for Gradio UI and multi-user callers.

Every browser session / user flow gets its own ``SessionState`` (with its own ``ProjectStore``,
``ApprovalQueue``, fresh ``today`` date, and ``asyncio.Lock``). Nothing mutable is shared across sessions,
and resetting a session regenerates sample task/deadline dates using ``date.today()`` (or an injected clock).

Optional restart durability is disabled by default (in-memory only) and enabled only when an explicit
``SQLiteSessionStore`` is configured via ``PMP_PERSIST_SESSIONS=true`` or passed explicitly.
"""
from __future__ import annotations

import asyncio
import copy
import secrets
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Optional

from data.sample_project import build_sample_store
from pipeline.approvals import ApprovalQueue, DecisionResult
from tools.execution_tools import ProjectStore, execute_action

if TYPE_CHECKING:
    from pipeline.persistence import SQLiteSessionStore


def _new_session_id() -> str:
    return secrets.token_hex(16)


@dataclass
class SessionState:
    session_id: str
    store: ProjectStore
    queue: ApprovalQueue
    today: date
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    epoch: int = 0
    _state_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    persistence: Optional[SQLiteSessionStore] = field(default=None, repr=False)

    def __deepcopy__(self, memo):
        with self._state_lock:
            return SessionState(
                session_id=self.session_id,
                store=copy.deepcopy(self.store, memo),
                queue=copy.deepcopy(self.queue, memo),
                today=copy.deepcopy(self.today, memo),
                lock=asyncio.Lock(),
                epoch=self.epoch,
                persistence=self.persistence,
            )


def persist_session(session: SessionState) -> None:
    """Persist ``session`` if durable persistence is attached (no-op in default in-memory mode)."""
    if session.persistence is not None:
        session.persistence.save_session(session)


def create_session(
    today: Optional[date] = None,
    persistence: Optional[SQLiteSessionStore] = None,
) -> SessionState:
    d = today or date.today()
    session = SessionState(
        session_id=_new_session_id(),
        store=build_sample_store(d),
        queue=ApprovalQueue(),
        today=d,
        persistence=persistence,
    )
    persist_session(session)
    return session


def reset_session(
    session: Optional[SessionState] = None,
    today: Optional[date] = None,
    persistence: Optional[SQLiteSessionStore] = None,
) -> SessionState:
    """Clear only this session's store & approval queue and regenerate sample data for ``today``."""
    d = today or date.today()
    if session is None:
        return create_session(d, persistence=persistence)
    with session._state_lock:
        backend = persistence if persistence is not None else session.persistence
        old_id = session.session_id
        session.epoch += 1
        session.session_id = _new_session_id()
        session.today = d
        session.store = build_sample_store(d)
        session.queue = ApprovalQueue()
        session.persistence = backend
        if backend is not None:
            backend.delete_session(old_id)
            backend.save_session(session)
        return session


def resolve_session_approval(
    session: SessionState,
    pid: str,
    approve: bool,
    note: str = "",
) -> DecisionResult:
    """Resolve an approval item on ``session``, coordinating durable commit & crash-recovery state if enabled."""
    backend = session.persistence
    if backend is None or not approve:
        res = session.queue.resolve(pid, approve, session.store, note)
        if backend is not None and res.applied:
            backend.save_session(session)
        return res

    cleaned_pid = (pid or "").strip()
    with session.queue._lock:
        item = session.queue._items.get(cleaned_pid)
        if item is None:
            return DecisionResult(ok=False, applied=False, message="Unknown approval id.")
        if item.status != "pending":
            return DecisionResult(
                ok=True,
                applied=False,
                item=item,
                message=f"Already decided ({item.status}); nothing was changed.",
            )
        # Ensure the pending record is persisted before marking it executing.
        backend.save_session(session)
        item.status = "executing"
        backend.mark_executing(session.session_id, item.id)

        result = execute_action(item.action, session.store, authorized=True)
        item.result = result
        item.status = "approved" if result.status == "confirmed" else "failed"
        item.decision_note = note
        item.decided_at = datetime.now(timezone.utc)
        backend.save_session(session)
        return DecisionResult(
            ok=True,
            applied=True,
            item=item,
            message=f"{item.status}: {result.detail}",
        )
