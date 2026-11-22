"""Step 7a - which tasks are overdue, blocked or planned beyond the project deadline. Pure Python, no LLM.

Two different things, kept apart on purpose:
* ``overdue``            - an open task whose own due date is before today.
* ``due_after_deadline`` - an open task whose due date is AFTER the project deadline (a schedule conflict,
                           even when the task is not late yet).
"""
from datetime import date
from typing import Optional

from pydantic import BaseModel

from domain.models import TaskItem


class TaskFindings(BaseModel):
    overdue: list[TaskItem]
    blocked: list[TaskItem]
    open_count: int
    done_count: int
    all_task_ids: list[str]
    task_catalog: list[str] = []
    due_after_deadline: list[TaskItem] = []
    deadline: Optional[date] = None


def check_tasks(tasks: list[TaskItem], today: Optional[date] = None, deadline: Optional[date] = None) -> TaskFindings:
    today = today or date.today()
    open_tasks = [t for t in tasks if t.status != "done"]
    overdue = [t for t in open_tasks if t.due_date is not None and t.due_date < today]
    blocked = [t for t in open_tasks if t.status == "blocked" or t.blocker]
    late_for_project = [t for t in open_tasks
                        if deadline is not None and t.due_date is not None and t.due_date > deadline]
    return TaskFindings(
        overdue=overdue,
        blocked=blocked,
        open_count=len(open_tasks),
        done_count=len(tasks) - len(open_tasks),
        all_task_ids=[t.id for t in tasks if t.id],
        task_catalog=[f"{t.id}: {t.title} ({t.status})" for t in tasks if t.id],
        due_after_deadline=late_for_project,
        deadline=deadline,
    )
