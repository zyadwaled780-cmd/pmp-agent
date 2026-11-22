"""Step 7b - who is overloaded / unavailable. Pure Python, no LLM.

capacity_per_day: person-days of work a person can finish per day (1.0 = full time).

The workload horizon is the number of days until the PROJECT DEADLINE (``deadline - today``). It is only
a fixed default (5 days) when no deadline is known, and the findings say which one was used.
Zero or negative horizons are explicit: no time is left, so any open work makes the owner overloaded.
"""
from datetime import date
from typing import Optional

from pydantic import BaseModel

from domain.models import TaskItem, estimate_of

DEFAULT_HORIZON_DAYS = 5.0


class TeamMember(BaseModel):
    name: str
    capacity_per_day: float = 1.0
    on_leave: bool = False


class PeopleFindings(BaseModel):
    workload_days: dict[str, float]      # open work (in days) per owner
    overloaded: list[str]
    unavailable: list[str]
    available: list[str]
    tasks_owned_by_unavailable: list[str]  # task titles
    unknown_owners: list[str]              # owners that are not in the team list
    member_names: list[str]
    horizon_days: float = DEFAULT_HORIZON_DAYS
    horizon_source: str = "default"        # deadline | explicit | default
    deadline_passed: bool = False


def resolve_horizon(horizon_days: Optional[float], deadline: Optional[date], today: Optional[date]) -> tuple[float, str, bool]:
    """(horizon in days, where it came from, deadline already passed?). Never negative."""
    if horizon_days is not None:
        return max(float(horizon_days), 0.0), "explicit", horizon_days < 0
    if deadline is not None:
        days = (deadline - (today or date.today())).days
        return float(max(days, 0)), "deadline", days < 0
    return DEFAULT_HORIZON_DAYS, "default", False


def check_team(
    members: list[TeamMember],
    tasks: list[TaskItem],
    horizon_days: Optional[float] = None,
    *,
    deadline: Optional[date] = None,
    today: Optional[date] = None,
) -> PeopleFindings:
    """A person is overloaded when open work exceeds capacity_per_day * horizon."""
    horizon, source, passed = resolve_horizon(horizon_days, deadline, today)
    by_name = {m.name: m for m in members}
    open_tasks = [t for t in tasks if t.status != "done"]

    workload: dict[str, float] = {m.name: 0.0 for m in members}
    unknown: set[str] = set()
    for t in open_tasks:
        if not t.owner:
            continue
        if t.owner in by_name:
            workload[t.owner] += estimate_of(t)
        else:
            unknown.add(t.owner)

    unavailable = [m.name for m in members if m.on_leave]
    overloaded = [
        m.name for m in members
        if not m.on_leave and workload[m.name] > m.capacity_per_day * horizon
    ]
    available = [m.name for m in members if not m.on_leave and m.name not in overloaded]
    stuck = [t.title for t in open_tasks if t.owner in unavailable]

    return PeopleFindings(
        workload_days=workload,
        overloaded=overloaded,
        unavailable=unavailable,
        available=available,
        tasks_owned_by_unavailable=stuck,
        unknown_owners=sorted(unknown),
        member_names=[m.name for m in members],
        horizon_days=horizon,
        horizon_source=source,
        deadline_passed=passed,
    )
