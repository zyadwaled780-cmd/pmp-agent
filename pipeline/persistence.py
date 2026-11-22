"""Optional, explicitly enabled durable SQLite persistence for isolated project sessions.

Design & security boundaries:
* Disabled by default; callers opt in explicitly via configuration (``PMP_PERSIST_SESSIONS=true``)
  or by injecting a ``SQLiteSessionStore`` instance.
* Strict session isolation: ``session_id`` values are opaque server-generated 32-char lowercase hex
  tokens validated against ``^[a-f0-9]{16,64}$`` and only ever passed as parameterized SQLite keys
  (never used to construct filesystem paths).
* Strict allowlist of persisted fields: only project tasks, team members, deadline, comments,
  budget state, task-ID high-water mark, session date, and manager approval records are stored.
  API keys, SDK clients, locks, and raw LLM conversation history are NEVER persisted.
* Transactional writes use ``BEGIN IMMEDIATE``. Before executing an approved action, callers may call
  ``mark_executing()`` so that an interrupted process crash leaves the item in ``executing`` on disk;
  reloading transitions ``executing`` -> ``interrupted`` without silent re-execution (SQLite alone
  cannot guarantee at-most-once execution across external side effects).
"""
from __future__ import annotations

import copy
import json
import re
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from checks.team_checker import TeamMember
from domain.models import ProposedAction, TaskItem
from pipeline.adapters import Route
from pipeline.approvals import ApprovalQueue, PendingApproval
from tools.execution_tools import ExecutionResult, ProjectStore

if TYPE_CHECKING:
    from pipeline.session import SessionState

SCHEMA_VERSION = 1
_SESSION_ID_RE = re.compile(r"^[a-f0-9]{16,64}$")
_TASK_ID_RE = re.compile(r"^T(\d+)$")


class PersistenceError(RuntimeError):
    """Base error for durable session persistence failures."""


class InvalidSessionIdError(PersistenceError, ValueError):
    """Session id is missing, malformed, or unsafe."""


class SchemaVersionError(PersistenceError):
    """SQLite database schema version does not match this code version."""


def validate_session_id(session_id: str) -> str:
    """Validate that ``session_id`` is an opaque lowercase hex token."""
    cleaned = session_id if isinstance(session_id, str) else ""
    if not _SESSION_ID_RE.fullmatch(cleaned):
        raise InvalidSessionIdError("session_id must be 16-64 lowercase hex characters")
    return cleaned


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _max_task_counter(tasks: dict[str, TaskItem], stored_counter: int) -> int:
    highest = max(0, int(stored_counter))
    for tid in tasks:
        m = _TASK_ID_RE.fullmatch(tid)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest


def _dump_model(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), separators=(",", ":"))


class SQLiteSessionStore:
    """Thread-safe SQLite persistence store for ``SessionState`` objects."""

    def __init__(self, db_path: Path | str):
        raw = str(db_path).strip()
        if not raw:
            raise PersistenceError("db_path must not be empty")
        self.db_path = Path(raw)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if str(self.db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY
                    )
                    """
                )
                row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
                if row is None:
                    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
                else:
                    found = int(row["version"])
                    if found != SCHEMA_VERSION:
                        conn.execute("ROLLBACK")
                        raise SchemaVersionError(
                            f"Unsupported session DB schema version {found} (expected {SCHEMA_VERSION})"
                        )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        today TEXT NOT NULL,
                        counter INTEGER NOT NULL,
                        deadline TEXT,
                        budget_used_percent REAL,
                        budget_history_json TEXT NOT NULL,
                        tasks_json TEXT NOT NULL,
                        members_json TEXT NOT NULL,
                        comments_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS approvals (
                        session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                        approval_id TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        action_json TEXT NOT NULL,
                        route_json TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        decided_at TEXT,
                        status TEXT NOT NULL,
                        result_json TEXT,
                        decision_note TEXT NOT NULL,
                        PRIMARY KEY (session_id, approval_id)
                    )
                    """
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()

    def save_session(self, session: SessionState) -> None:
        """Atomically persist project state and approval records for ``session``."""
        sid = validate_session_id(session.session_id)
        with session.store.lock:
            tasks_payload = {tid: t.model_dump(mode="json") for tid, t in session.store.tasks.items()}
            members_payload = [m.model_dump(mode="json") for m in session.store.members]
            deadline_val = session.store.deadline.isoformat() if session.store.deadline else None
            budget_used = session.store.budget_used_percent
            budget_hist = json.dumps(copy.deepcopy(session.store.budget_history), separators=(",", ":"))
            comments_payload = json.dumps(copy.deepcopy(session.store.comments), separators=(",", ":"))
            counter = _max_task_counter(session.store.tasks, session.store._counter)
            today_val = session.today.isoformat()

        approvals = session.queue.snapshot()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, today, counter, deadline, budget_used_percent,
                        budget_history_json, tasks_json, members_json, comments_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        today = excluded.today,
                        counter = excluded.counter,
                        deadline = excluded.deadline,
                        budget_used_percent = excluded.budget_used_percent,
                        budget_history_json = excluded.budget_history_json,
                        tasks_json = excluded.tasks_json,
                        members_json = excluded.members_json,
                        comments_json = excluded.comments_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        sid,
                        today_val,
                        counter,
                        deadline_val,
                        budget_used,
                        budget_hist,
                        json.dumps(tasks_payload, separators=(",", ":")),
                        json.dumps(members_payload, separators=(",", ":")),
                        comments_payload,
                        _now_iso(),
                    ),
                )
                conn.execute("DELETE FROM approvals WHERE session_id = ?", (sid,))
                for seq, item in enumerate(approvals):
                    conn.execute(
                        """
                        INSERT INTO approvals (
                            session_id, approval_id, seq, action_json, route_json,
                            reason, created_at, decided_at, status, result_json, decision_note
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            sid,
                            item.id,
                            seq,
                            _dump_model(item.action),
                            json.dumps(item.route.value),
                            item.reason,
                            item.created_at.isoformat(),
                            item.decided_at.isoformat() if item.decided_at else None,
                            item.status,
                            _dump_model(item.result) if item.result is not None else None,
                            item.decision_note,
                        ),
                    )
                conn.execute("COMMIT")
            except Exception:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()

    def mark_executing(self, session_id: str, approval_id: str) -> None:
        """Persist an in-flight ``executing`` status before mutating external/project state."""
        sid = validate_session_id(session_id)
        pid = (approval_id or "").strip()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    """
                    UPDATE approvals
                    SET status = 'executing'
                    WHERE session_id = ? AND approval_id = ? AND status = 'pending'
                    """,
                    (sid, pid),
                )
                if cur.rowcount == 0:
                    row = conn.execute(
                        "SELECT status FROM approvals WHERE session_id = ? AND approval_id = ?",
                        (sid, pid),
                    ).fetchone()
                    if row is None:
                        raise PersistenceError(f"Approval record '{pid}' missing for persisted session")
                conn.execute("COMMIT")
            except Exception:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()

    def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load a persisted session, transitioning any crashed ``executing`` approval to ``interrupted``."""
        from pipeline.session import SessionState

        sid = validate_session_id(session_id)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                srow = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (sid,),
                ).fetchone()
                if srow is None:
                    conn.execute("COMMIT")
                    return None

                arows = conn.execute(
                    "SELECT * FROM approvals WHERE session_id = ? ORDER BY seq ASC",
                    (sid,),
                ).fetchall()

                interrupted_note = (
                    "Interrupted mid-execution before durable commit; verify project state manually "
                    "before retrying or creating a follow-up action."
                )
                items: list[PendingApproval] = []
                recovered_any = False
                now_dt = datetime.now(timezone.utc)
                for row in arows:
                    status = str(row["status"])
                    decided_raw = row["decided_at"]
                    decided_at = datetime.fromisoformat(decided_raw) if decided_raw else None
                    decision_note = str(row["decision_note"])
                    if status == "executing":
                        status = "interrupted"
                        decided_at = decided_at or now_dt
                        decision_note = decision_note or interrupted_note
                        recovered_any = True
                        conn.execute(
                            """
                            UPDATE approvals
                            SET status = ?, decided_at = ?, decision_note = ?
                            WHERE session_id = ? AND approval_id = ?
                            """,
                            (status, decided_at.isoformat(), decision_note, sid, str(row["approval_id"])),
                        )

                    result_raw = row["result_json"]
                    items.append(
                        PendingApproval(
                            id=str(row["approval_id"]),
                            action=ProposedAction.model_validate_json(row["action_json"]),
                            route=Route(json.loads(str(row["route_json"]))),
                            reason=str(row["reason"]),
                            created_at=datetime.fromisoformat(str(row["created_at"])),
                            decided_at=decided_at,
                            status=status,  # type: ignore[arg-type]
                            result=ExecutionResult.model_validate_json(result_raw) if result_raw else None,
                            decision_note=decision_note,
                        )
                    )
                conn.execute("COMMIT")
            except Exception:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()

        tasks_raw: dict[str, Any] = json.loads(str(srow["tasks_json"]))
        tasks = {tid: TaskItem.model_validate(payload) for tid, payload in tasks_raw.items()}
        members = [TeamMember.model_validate(m) for m in json.loads(str(srow["members_json"]))]
        deadline = date.fromisoformat(str(srow["deadline"])) if srow["deadline"] else None
        store = ProjectStore(tasks=list(tasks.values()), members=members, deadline=deadline)
        store.comments = json.loads(str(srow["comments_json"]))
        store.budget_used_percent = srow["budget_used_percent"]
        store.budget_history = json.loads(str(srow["budget_history_json"]))
        store._counter = _max_task_counter(tasks, int(srow["counter"]))

        queue = ApprovalQueue()
        queue.restore_from(items)
        session = SessionState(
            session_id=sid,
            store=store,
            queue=queue,
            today=date.fromisoformat(str(srow["today"])),
            persistence=self,
        )
        if recovered_any:
            # Ensure session row reflects the clean recovery timestamp without altering store contents.
            self.save_session(session)
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete only ``session_id`` and its approvals from SQLite."""
        sid = validate_session_id(session_id)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
                conn.execute("COMMIT")
                return cur.rowcount > 0
            except Exception:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()

    def list_session_ids(self) -> list[str]:
        """Return persisted session IDs ordered by most recently updated."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT session_id FROM sessions ORDER BY updated_at DESC").fetchall()
                return [str(r["session_id"]) for r in rows]
            finally:
                conn.close()
