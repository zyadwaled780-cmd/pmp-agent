"""Step 7 - run the checks in parallel; one failing check never kills the run."""
import asyncio
from datetime import date
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from checks.budget_checker import BudgetFindings, check_budget
from checks.deadline_checker import DeadlineFindings, check_deadline
from checks.task_checker import TaskFindings, check_tasks
from checks.team_checker import PeopleFindings, TeamMember, check_team
from domain.models import TaskItem


class CheckResults(BaseModel):
    tasks: Optional[TaskFindings] = None
    people: Optional[PeopleFindings] = None
    deadline: Optional[DeadlineFindings] = None
    budget: Optional[BudgetFindings] = None
    data_gaps: list[str] = Field(default_factory=list)

    def known_task_ids(self) -> set[str]:
        return set(self.tasks.all_task_ids) if self.tasks else set()

    def known_people(self) -> set[str]:
        return set(self.people.member_names) if self.people else set()


ALL_WORKERS = ("tasks", "people", "deadline")


def _team(members, tasks, deadline, today):
    return check_team(members, tasks, None, deadline=deadline, today=today)


async def run_all_checks(
    tasks: list[TaskItem],
    members: list[TeamMember],
    deadline: Optional[date],
    today: Optional[date] = None,
    workers: tuple[str, ...] = ALL_WORKERS,
    budget_used_percent: Optional[float] = None,
) -> CheckResults:
    """Run the chosen checks at the same time. A worker that was not chosen stays None
    (skipped on purpose, no data gap); a worker that crashed stays None WITH a data gap.
    The budget check is cheap and deterministic: it runs whenever a budget value is known."""
    jobs: dict[str, tuple[str, Callable[..., Any], tuple[Any, ...]]] = {
        "tasks": ("task check", check_tasks, (tasks, today, deadline)),
        "people": ("team check", _team, (members, tasks, deadline, today)),
        "deadline": ("deadline check", check_deadline, (tasks, members, deadline, today)),
    }
    chosen = [w for w in ALL_WORKERS if w in workers]
    outcomes = await asyncio.gather(
        *[asyncio.to_thread(jobs[w][1], *jobs[w][2]) for w in chosen], return_exceptions=True
    )
    results = CheckResults()
    for w, out in zip(chosen, outcomes, strict=True):
        if isinstance(out, Exception):
            results.data_gaps.append(f"{jobs[w][0]} unavailable: {type(out).__name__}: {out}")
        else:
            setattr(results, w, out)
    if budget_used_percent is not None:
        try:
            results.budget = check_budget(budget_used_percent)
        except Exception as exc:
            results.data_gaps.append(f"budget check unavailable: {type(exc).__name__}: {exc}")
    return results
