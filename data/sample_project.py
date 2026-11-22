"""Demo project. Dates are relative to today so the demo never goes stale."""
from datetime import date, timedelta
from typing import Optional

from checks.team_checker import TeamMember
from domain.models import TaskItem
from tools.execution_tools import ProjectStore


def build_sample_store(today: Optional[date] = None) -> ProjectStore:
    today = today or date.today()

    def d(n: int) -> date:
        return today + timedelta(days=n)
    tasks = [
        TaskItem(id="T1", title="Login page", owner="Ahmed", status="done", due_date=d(-5), estimate_days=2),
        TaskItem(id="T2", title="Payment API", owner="Sara", status="blocked", due_date=d(-2), estimate_days=4,
                 blocker="Waiting for the supplier's sandbox keys"),
        TaskItem(id="T3", title="Reports dashboard", owner="Sara", status="in_progress", due_date=d(9), estimate_days=4),
        TaskItem(id="T4", title="Documentation", owner="Omar", status="todo", due_date=d(-1), estimate_days=1),
        TaskItem(id="T5", title="Mobile layout", owner="Ahmed", status="in_progress", due_date=d(6), estimate_days=3),
    ]
    members = [TeamMember(name="Ahmed"), TeamMember(name="Sara"),
               TeamMember(name="Omar", on_leave=True), TeamMember(name="Mona")]
    return ProjectStore(tasks, members, deadline=d(4))
