"""The run context handed to every SDK tool (RunContextWrapper[PMPContext])."""
from dataclasses import dataclass, field
from datetime import date

from pipeline.adapters import Route
from pipeline.approvals import ApprovalQueue
from tools.execution_tools import ProjectStore


@dataclass
class PMPContext:
    store: ProjectStore
    queue: ApprovalQueue
    project_id: str = "demo"
    user_role: str = "manager"          # who is acting: only manager / pm may change the project
    today: date = field(default_factory=date.today)
    route: Route = Route.NORMAL         # set by the router, read by the action gate
    audit: list[str] = field(default_factory=list)
