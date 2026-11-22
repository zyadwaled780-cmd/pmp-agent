"""Agent-layer models (what the LLM steps produce and the checks consume).

The partner's core/models.py is the policy-layer contract; pipeline/adapters.py translates
between the two. Models that an LLM must fill (ExtractedUpdate, ProposedAction, RiskItem ...)
avoid free-form dicts, because the Agents SDK strict schema rejects them.
"""
from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    SUPPLIER = "supplier"
    EMPLOYEE = "employee"
    ACCOUNTANT = "accountant"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_ORDER = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}

TaskStatus = Literal["todo", "in_progress", "blocked", "done"]
ActionName = Literal[
    "create_task", "update_task", "assign_task", "create_project_comment", "delete_task"
]


class InputMessage(BaseModel):
    source: SourceType
    sender: str
    text: str
    received_at: Optional[datetime] = None


DEFAULT_ESTIMATE_DAYS = 1.0


class TaskItem(BaseModel):
    id: Optional[str] = None
    title: str
    owner: Optional[str] = None
    status: TaskStatus
    due_date: Optional[date] = None
    estimate_days: Optional[float] = None  # None means "unknown", treated as DEFAULT_ESTIMATE_DAYS
    blocker: Optional[str] = None
    delay_days: Optional[int] = None   # days of delay the message states for this task


def estimate_of(task: "TaskItem") -> float:
    return DEFAULT_ESTIMATE_DAYS if task.estimate_days is None else task.estimate_days


class ExtractedUpdate(BaseModel):
    source: SourceType
    sender: str
    summary: str
    tasks: list[TaskItem]
    blockers: list[str]
    risks: list[str]
    mentioned_deadline: Optional[date] = None
    budget_used_percent: Optional[float] = None


class RiskItem(BaseModel):
    title: str
    severity: Severity
    evidence: str
    related_task_ids: list[str]


class ProposedAction(BaseModel):
    action: ActionName
    task_id: Optional[str] = None
    title: Optional[str] = None
    owner: Optional[str] = None
    status: Optional[TaskStatus] = None
    due_date: Optional[date] = None
    comment: Optional[str] = None
    reason: str
    evidence: str


class ExecutiveReport(BaseModel):
    """Field ownership (who may change what):
    * summary, proposed_actions -> Status Writer (the only fields the writer can revise)
    * risks, model_risk_titles  -> Risk Analyzer (deterministic risks are computed from checks and are
                                   authoritative; model-generated ones must be grounded in the facts)
    * data_gaps, review_status  -> the pipeline
    """
    summary: str
    risks: list[RiskItem]
    proposed_actions: list[ProposedAction]
    data_gaps: list[str] = Field(default_factory=list)
    model_risk_titles: list[str] = Field(default_factory=list)   # titles of risks the LLM added (not deterministic)
    review_status: Literal["approved", "needs_review", "withheld"] = "approved"
