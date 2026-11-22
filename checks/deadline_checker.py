"""Step 7c (and step 10) - deadline risk by arithmetic, never by guessing.

days_left is counted in calendar days and team capacity is person-days per day.
If you want working days only, scale capacity_per_day accordingly.
`critical` is the flag the pipeline uses to raise the CRITICAL path / manager alert.
"""
from datetime import date
from enum import Enum
from math import ceil
from typing import Optional

from pydantic import BaseModel

from checks.team_checker import TeamMember
from domain.models import TaskItem, estimate_of


class DeadlineStatus(str, Enum):
    ON_TRACK = "on_track"
    WARNING = "warning"
    AT_RISK = "at_risk"
    UNKNOWN = "unknown"


class DeadlineFindings(BaseModel):
    deadline: Optional[date]
    days_left: Optional[int]
    remaining_work_days: float
    team_capacity_per_day: float
    days_needed: Optional[float]
    slip_days: int
    status: DeadlineStatus
    critical: bool
    reason: str


def check_deadline(
    tasks: list[TaskItem],
    members: list[TeamMember],
    deadline: Optional[date],
    today: Optional[date] = None,
    warning_ratio: float = 0.8,
) -> DeadlineFindings:
    today = today or date.today()
    remaining = sum(estimate_of(t) for t in tasks if t.status != "done")
    capacity = sum(m.capacity_per_day for m in members if not m.on_leave)

    def build(status, reason, days_left=None, needed=None, slip=0):
        return DeadlineFindings(
            deadline=deadline, days_left=days_left, remaining_work_days=remaining,
            team_capacity_per_day=capacity, days_needed=needed, slip_days=slip,
            status=status, critical=(status == DeadlineStatus.AT_RISK), reason=reason,
        )

    if deadline is None:
        return build(DeadlineStatus.UNKNOWN, "No project deadline is set, risk cannot be calculated.")

    days_left = (deadline - today).days
    if remaining == 0:
        return build(DeadlineStatus.ON_TRACK, "No open work remains.", days_left, 0.0)
    if days_left < 0:
        return build(
            DeadlineStatus.AT_RISK,
            f"Deadline {deadline} passed {-days_left} day(s) ago with {remaining:g} day(s) of work open.",
            days_left, None, -days_left,
        )
    if capacity <= 0:
        return build(
            DeadlineStatus.AT_RISK,
            f"No available capacity for {remaining:g} day(s) of open work, {days_left} day(s) left.",
            days_left, None, 0,
        )

    needed = remaining / capacity
    if needed > days_left:
        slip = ceil(needed - days_left)
        return build(
            DeadlineStatus.AT_RISK,
            f"Open work needs {needed:.1f} day(s) at current capacity but only {days_left} day(s) are left (slip about {slip}).",
            days_left, needed, slip,
        )
    if needed > days_left * warning_ratio:
        return build(
            DeadlineStatus.WARNING,
            f"Open work needs {needed:.1f} of the {days_left} day(s) left, very little buffer.",
            days_left, needed,
        )
    return build(
        DeadlineStatus.ON_TRACK,
        f"Open work needs {needed:.1f} of the {days_left} day(s) left.",
        days_left, needed,
    )
