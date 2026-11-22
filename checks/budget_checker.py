"""Budget check (deterministic). The reported percentage of budget used, judged by fixed thresholds.

Semantics of budget state (see pipeline/budget.py): the LATEST valid report from an accountant replaces the
stored value; reports from other roles are used for the current turn only and are never persisted.
"""
import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel

MAX_BUDGET_PERCENT = 1000.0     # sanity bound: a budget more than 10x exceeded is a data error


class BudgetStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"       # >= 80 %
    CRITICAL = "critical"     # >= 100 %
    UNKNOWN = "unknown"


class BudgetFindings(BaseModel):
    used_percent: Optional[float]
    status: BudgetStatus
    critical: bool
    reason: str


def validate_budget_percent(value) -> float:
    """Return a clean float percentage or raise ValueError (bool, NaN, inf, negative, absurd values)."""
    if isinstance(value, bool) or value is None:
        raise ValueError("budget percentage must be a number")
    try:
        pct = float(value)
    except (TypeError, ValueError):
        raise ValueError("budget percentage must be a number") from None
    if math.isnan(pct) or math.isinf(pct):
        raise ValueError("budget percentage must be finite")
    if pct < 0 or pct > MAX_BUDGET_PERCENT:
        raise ValueError(f"budget percentage must be between 0 and {MAX_BUDGET_PERCENT:g}")
    return pct


def check_budget(used_percent: Optional[float]) -> BudgetFindings:
    if used_percent is None:
        return BudgetFindings(used_percent=None, status=BudgetStatus.UNKNOWN, critical=False,
                              reason="No budget usage has been reported.")
    pct = validate_budget_percent(used_percent)
    if pct >= 100:
        return BudgetFindings(used_percent=pct, status=BudgetStatus.CRITICAL, critical=True,
                              reason=f"{pct:g}% of the budget is used: the budget is exceeded.")
    if pct >= 80:
        return BudgetFindings(used_percent=pct, status=BudgetStatus.WARNING, critical=False,
                              reason=f"{pct:g}% of the budget is used.")
    return BudgetFindings(used_percent=pct, status=BudgetStatus.OK, critical=False,
                          reason=f"{pct:g}% of the budget is used.")
