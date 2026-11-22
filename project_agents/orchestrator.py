"""Step 6 - the Project Orchestrator (LLM-orchestrated): which checks does THIS request need?

The model proposes a plan; Python validates it. Unknown workers are dropped, an empty or
failed plan means "run everything", and the deadline check is forced when something blocks delivery.
"""
from typing import Literal, Optional, cast

from pydantic import BaseModel

from checks import ALL_WORKERS
from domain.models import ExtractedUpdate
from project_agents.llm import LLMCall, call_llm

Worker = Literal["tasks", "people", "deadline"]


class OrchestrationPlan(BaseModel):
    workers: list[Worker]
    focus: str


INSTRUCTIONS = """You are the project orchestrator. Pick which checks are needed to answer this request:
- tasks: overdue and blocked tasks
- people: who is overloaded or on leave
- deadline: days left versus remaining work
Pick only what the request needs (a question about workload may need only tasks and people).
When unsure, pick all three. focus: one short sentence on what the report should concentrate on."""


async def plan_work(
    update: ExtractedUpdate,
    original_text: str,
    force_deadline: bool = False,
    llm: Optional[LLMCall] = call_llm,
) -> tuple[OrchestrationPlan, Optional[str]]:
    """Returns (validated plan, note). The note explains any fallback."""
    note: Optional[str] = None
    plan: Optional[OrchestrationPlan] = None
    if llm is not None:
        prompt = f"<request>\n{original_text}\n</request>\nsummary: {update.summary}\nblockers: {update.blockers}"
        try:
            plan = await llm("Project Orchestrator", INSTRUCTIONS, prompt, OrchestrationPlan)
        except Exception as exc:
            note = f"Orchestrator unavailable ({type(exc).__name__}), running all checks."
    workers: list[Worker] = [cast(Worker, w) for w in ALL_WORKERS if plan and w in plan.workers]
    if not workers:
        workers = [cast(Worker, w) for w in ALL_WORKERS]
        note = note or "Empty plan, running all checks."
    if force_deadline and "deadline" not in workers:
        workers.append("deadline")
        note = (note + " " if note else "") + "Deadline check added because something blocks delivery."
    return OrchestrationPlan(workers=workers, focus=plan.focus if plan else "full project status"), note
