"""
Core Data Models for PMP Multi-Agent AI System
Based on the architecture specifications in PMP_Agent_Process_AR / EN.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UpdateSource(str, Enum):
    SUPPLIER = "SUPPLIER"      # المورد (توريد، شحنات، تأخير)
    EMPLOYEE = "EMPLOYEE"      # الموظف (تقدم المهام، عوائق)
    ACCOUNTANT = "ACCOUNTANT"  # المحاسب (ميزانية، تكلفة)
    UNKNOWN = "UNKNOWN"


class InputMessage(BaseModel):
    message_id: str
    source: UpdateSource
    sender_id: str
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SecurityAssessment(BaseModel):
    is_safe: bool
    threat_type: Optional[str] = None  # e.g., "PROMPT_INJECTION", "MALICIOUS_COMMAND"
    reason: Optional[str] = None
    quarantined: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class TaskItem(BaseModel):
    task_id: str
    title: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    status: str = "IN_PROGRESS"  # CREATED, IN_PROGRESS, BLOCKED, COMPLETED
    is_blocker: bool = False
    delay_days: int = 0
    notes: Optional[str] = None


class ExtractedUpdate(BaseModel):
    update_id: str
    source: UpdateSource
    tasks: List[TaskItem] = Field(default_factory=list)
    raw_entities: Dict[str, Any] = Field(default_factory=dict)
    cost_budget_impact: Optional[str] = None
    is_complete: bool = True
    missing_fields: List[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None


class RouteType(str, Enum):
    NORMAL = "NORMAL"                # مسار عادي
    CRITICAL_PATH = "CRITICAL_PATH"  # مسار حرج (تنبيه فوري للمدير + موافقة بشرية)


class RouteDecision(BaseModel):
    route: RouteType
    requires_immediate_alert: bool = False
    reason: str


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskItem(BaseModel):
    risk_id: str
    category: str  # e.g. "SUPPLY_CHAIN", "SCHEDULE", "BUDGET", "TECHNICAL"
    description: str
    severity: RiskLevel
    affected_tasks: List[str] = Field(default_factory=list)
    mitigation_suggestion: Optional[str] = None


class ActionType(str, Enum):
    ADD_COMMENT = "ADD_COMMENT"              # إضافة تعليق (مسموح)
    CREATE_TASK = "CREATE_TASK"              # إنشاء مهمة جديدة
    UPDATE_DUE_DATE = "UPDATE_DUE_DATE"      # تعديل موعد (يحتاج موافقة غالباً)
    REASSIGN_TASK = "REASSIGN_TASK"          # إعادة إسناد (يحتاج موافقة)
    DELETE_TASK = "DELETE_TASK"              # حذف مهمة (ممنوع دائماً)
    CUSTOM_API = "CUSTOM_API"


class ActionPolicyVerdict(str, Enum):
    ALLOWED = "ALLOWED"                      # مسموح بالتنفيذ التلقائي
    NEEDS_APPROVAL = "NEEDS_APPROVAL"        # يتطلب موافقة المدير البشري
    FORBIDDEN = "FORBIDDEN"                  # ممنوع قطعي ومرفوض


class ProposedAction(BaseModel):
    action_id: str
    action_type: ActionType
    target: str                              # Task ID or Project Entity
    payload: Dict[str, Any] = Field(default_factory=dict)
    justification: str
    verdict: Optional[ActionPolicyVerdict] = None
    verdict_reason: Optional[str] = None
    approved_by_manager: Optional[bool] = None
    executed: bool = False
    execution_result: Optional[str] = None


class ExecutiveReport(BaseModel):
    report_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str
    tasks_summary: List[str] = Field(default_factory=list)
    team_workload: List[str] = Field(default_factory=list)
    risks: List[RiskItem] = Field(default_factory=list)
    proposed_actions: List[ProposedAction] = Field(default_factory=list)
    is_verified_by_evaluator: bool = False
    audit_trace: List[str] = Field(default_factory=list)
