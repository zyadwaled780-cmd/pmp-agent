"""The project read tools. They acquire store.lock and return copies so callers cannot mutate the store."""
from datetime import date
from typing import Optional

from checks.team_checker import TeamMember
from domain.models import TaskItem
from tools.execution_tools import ProjectStore


def get_project_tasks(store: ProjectStore) -> list[TaskItem]:
    with store.lock:
        return [t.model_copy(deep=True) for t in store.tasks.values()]


def get_task(store: ProjectStore, task_id: str) -> Optional[TaskItem]:
    with store.lock:
        t = store.tasks.get(task_id)
        return t.model_copy(deep=True) if t else None


def get_team_members(store: ProjectStore) -> list[TeamMember]:
    with store.lock:
        return [m.model_copy(deep=True) for m in store.members]


def get_project_deadline(store: ProjectStore) -> Optional[date]:
    with store.lock:
        return store.deadline


def get_project_comments(store: ProjectStore) -> list[dict]:
    with store.lock:
        return [dict(c) for c in store.comments]


def get_project_budget(store: ProjectStore) -> Optional[float]:
    with store.lock:
        return store.budget_used_percent
