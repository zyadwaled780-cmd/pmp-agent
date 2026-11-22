"""The ONE authoritative validator for every project-changing action.

It runs (a) before an action is queued or executed by the pipeline / SDK write tools and
(b) AGAIN, under the store lock, immediately before execution (``tools.execution_tools.execute_action``),
so an approval that was queued earlier is checked against the CURRENT state, not the state at proposal time.
Writer-side filtering is only an optimisation; this is the rule set that counts.

Returns ``None`` when the action is valid, otherwise a short sanitized error text. It never mutates anything.
"""
from typing import Optional

from domain.models import ProposedAction

KNOWN_ACTIONS = ("create_task", "update_task", "assign_task", "create_project_comment")


def _owner_error(owner: Optional[str], store) -> Optional[str]:
    """An owner must be a real, available team member (an empty team list means 'unknown roster': allowed)."""
    if owner is None:
        return None
    if not owner.strip():
        return "owner must not be blank"
    names = {m.name: m for m in store.members}
    if names and owner not in names:
        return f"'{owner}' is not in the team"
    if owner in names and names[owner].on_leave:
        return f"'{owner}' is on leave"
    return None


def validate_action(a: ProposedAction, store) -> Optional[str]:
    kind = a.action
    if kind == "delete_task":
        return "delete_task is forbidden"
    if kind not in KNOWN_ACTIONS:
        return f"unknown action '{kind}'"

    if kind in ("update_task", "assign_task") and not a.task_id:
        return "task_id is required"
    if a.task_id and a.task_id not in store.tasks:
        return f"unknown task '{a.task_id}'"
    current = store.tasks.get(a.task_id) if a.task_id else None

    if kind == "create_task":
        if not (a.title or "").strip():
            return "title is required"
        return _owner_error(a.owner, store)          # None owner = intentionally unassigned task

    if kind == "assign_task":
        if a.owner is None or not a.owner.strip():
            return "owner is required"
        err = _owner_error(a.owner, store)
        if err:
            return err
        if current is not None and current.owner == a.owner:
            return f"task '{a.task_id}' is already assigned to '{a.owner}' (redundant assignment)"
        return None

    if kind == "update_task":
        if current is None:
            return f"unknown task '{a.task_id}'"
        if a.status is None and a.due_date is None:
            return "nothing to update"
        status_same = a.status is None or a.status == current.status
        due_same = a.due_date is None or a.due_date == current.due_date
        if status_same and due_same:
            what = []
            if a.status is not None:
                what.append(f"status '{current.status}'")
            if a.due_date is not None:
                what.append(f"due date {current.due_date}")
            return f"task '{a.task_id}' already has {' and '.join(what)} (redundant update)"
        return None

    # create_project_comment
    if not (a.comment or "").strip():
        return "comment text is required"
    return None
