"""The project tools as OpenAI Agents SDK function_tools.

READ tools (4 from the brief) + CALC tools (3, deterministic Python) are used by the worker agents.
WRITE tools (4 from the brief) go through pipeline.actions.apply_action, so they are protected by:
  1. a tool-input guardrail (only a manager / pm role may change the project),
  2. argument validation, 3. the action gate, 4. the manager approval queue.
Every tool receives the run context (RunContextWrapper[PMPContext]) and returns plain text/JSON.
Project data is scanned for hidden instructions before the model sees it.
"""
import dataclasses
import json
from datetime import date
from typing import Optional, cast

from agents import (
    RunContextWrapper,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    function_tool,
    tool_input_guardrail,
)

from checks.deadline_checker import check_deadline
from checks.task_checker import check_tasks
from checks.team_checker import check_team
from domain.models import ProposedAction, TaskStatus
from pipeline.actions import apply_action
from pipeline.adapters import scan_members, scan_tasks
from pipeline.context import PMPContext
from tools.read_tools import get_project_deadline, get_project_tasks, get_team_members

CHANGE_ROLES = {"manager", "pm"}


def _tasks(ctx: RunContextWrapper[PMPContext]):
    tasks, notes = scan_tasks(get_project_tasks(ctx.context.store))
    ctx.context.audit.extend(notes)
    return tasks


def _members(ctx: RunContextWrapper[PMPContext]):
    members, notes = scan_members(get_team_members(ctx.context.store))
    ctx.context.audit.extend(notes)
    return members


def clone_tools(tools: list, *, strict: Optional[bool] = None) -> list:
    """Return local copies of FunctionTool objects without mutating module-level globals."""
    cloned = []
    for t in tools:
        if strict is not None and hasattr(t, "strict_json_schema"):
            cloned.append(dataclasses.replace(t, strict_json_schema=strict))
        else:
            cloned.append(dataclasses.replace(t))
    return cloned


# ---------------- READ tools ----------------
@function_tool(name_override="get_project_tasks")
def get_project_tasks_tool(ctx: RunContextWrapper[PMPContext]) -> str:
    """List every project task: id, title, owner, status, due date, estimate in days and blocker."""
    ctx.context.audit.append("get_project_tasks")
    return json.dumps([t.model_dump(mode="json") for t in _tasks(ctx)], ensure_ascii=False)


@function_tool(name_override="get_task")
def get_task_tool(ctx: RunContextWrapper[PMPContext], task_id: str) -> str:
    """Get one task by id. Returns an error text when the id does not exist."""
    ctx.context.audit.append(f"get_task {task_id}")
    match = [t for t in _tasks(ctx) if t.id == task_id]
    return match[0].model_dump_json() if match else f"ERROR: unknown task '{task_id}'"


@function_tool(name_override="get_team_members")
def get_team_members_tool(ctx: RunContextWrapper[PMPContext]) -> str:
    """List the team: name, capacity per day (1.0 = full time) and whether the person is on leave."""
    ctx.context.audit.append("get_team_members")
    return json.dumps([m.model_dump() for m in _members(ctx)], ensure_ascii=False)


@function_tool(name_override="get_project_deadline")
def get_project_deadline_tool(ctx: RunContextWrapper[PMPContext]) -> str:
    """The project's final deadline (ISO date) or 'none'."""
    ctx.context.audit.append("get_project_deadline")
    d = get_project_deadline(ctx.context.store)
    return d.isoformat() if d else "none"


# ---------------- CALC tools (Python does the arithmetic) ----------------
@function_tool
def find_overdue_and_blocked(ctx: RunContextWrapper[PMPContext]) -> str:
    """Deterministic check: which open tasks are overdue, blocked, or due after the project deadline."""
    ctx.context.audit.append("find_overdue_and_blocked")
    c = ctx.context
    return check_tasks(_tasks(ctx), c.today, get_project_deadline(c.store)).model_dump_json()


@function_tool
def calculate_team_workload(ctx: RunContextWrapper[PMPContext]) -> str:
    """Deterministic check: open work per person, who is overloaded, who is on leave."""
    ctx.context.audit.append("calculate_team_workload")
    c = ctx.context
    return check_team(_members(ctx), _tasks(ctx), deadline=get_project_deadline(c.store), today=c.today).model_dump_json()


@function_tool
def calculate_deadline_risk(ctx: RunContextWrapper[PMPContext]) -> str:
    """Deterministic check: days left versus remaining work at the team's current capacity."""
    ctx.context.audit.append("calculate_deadline_risk")
    c = ctx.context
    return check_deadline(_tasks(ctx), _members(ctx), get_project_deadline(c.store), c.today).model_dump_json()


# ---------------- WRITE tools (protected) ----------------
@tool_input_guardrail
def only_managers_may_change(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Tool-level safety: the acting role is checked before ANY write tool runs."""
    role = data.context.context.user_role
    if role not in CHANGE_ROLES:
        return ToolGuardrailFunctionOutput.reject_content(
            f"BLOCKED: role '{role}' may not change the project. Nothing was executed.",
            output_info={"role": role})
    return ToolGuardrailFunctionOutput.allow()


def _outcome_text(o) -> str:
    if o.status == "executed":
        return f"CONFIRMED: {o.detail}"
    if o.status == "queued":
        return f"QUEUED for manager approval (id {o.approval_id}): {o.detail}. NOT executed yet."
    if o.status == "blocked":
        return f"BLOCKED: {o.detail}. Nothing was executed."
    return f"FAILED: {o.detail}. Do not report this as done."


def _parse_date(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


def _run(ctx, action: ProposedAction) -> str:
    return _outcome_text(apply_action(action, ctx.context))


@function_tool(tool_input_guardrails=[only_managers_may_change])
def create_task(ctx: RunContextWrapper[PMPContext], title: str, owner: Optional[str], reason: str,
                due_date: Optional[str] = None) -> str:
    """Create a new task. Needs a short reason and an optional due date (YYYY-MM-DD)."""
    try:
        due = _parse_date(due_date)
    except ValueError:
        return f"BLOCKED: invalid date '{due_date}'. Nothing was executed."
    return _run(ctx, ProposedAction(action="create_task", title=title, owner=owner, due_date=due,
                                    reason=reason, evidence=reason))


@function_tool(tool_input_guardrails=[only_managers_may_change])
def update_task(ctx: RunContextWrapper[PMPContext], task_id: str, status: Optional[str],
                due_date: Optional[str], reason: str) -> str:
    """Change a task's status (todo, in_progress, blocked, done) and/or due date (YYYY-MM-DD)."""
    if status is not None and status not in ("todo", "in_progress", "blocked", "done"):
        return f"BLOCKED: invalid status '{status}'. Nothing was executed."
    typed_status = cast(Optional[TaskStatus], status)
    try:
        due = _parse_date(due_date)
    except ValueError:
        return f"BLOCKED: invalid date '{due_date}'. Nothing was executed."
    return _run(ctx, ProposedAction(action="update_task", task_id=task_id, status=typed_status, due_date=due,
                                    reason=reason, evidence=reason))


@function_tool(tool_input_guardrails=[only_managers_may_change])
def assign_task(ctx: RunContextWrapper[PMPContext], task_id: str, owner: str, reason: str) -> str:
    """Assign a task to a team member. Always needs the manager's approval."""
    return _run(ctx, ProposedAction(action="assign_task", task_id=task_id, owner=owner, reason=reason, evidence=reason))


@function_tool(tool_input_guardrails=[only_managers_may_change])
def create_project_comment(ctx: RunContextWrapper[PMPContext], comment: str, task_id: Optional[str]) -> str:
    """Add a comment to the project record (optionally on one task)."""
    return _run(ctx, ProposedAction(action="create_project_comment", task_id=task_id, comment=comment,
                                    reason="comment", evidence=comment))


READ_TOOLS = [get_project_tasks_tool, get_task_tool, get_team_members_tool, get_project_deadline_tool]
CALC_TOOLS = [find_overdue_and_blocked, calculate_team_workload, calculate_deadline_risk]
WRITE_TOOLS = [create_task, update_task, assign_task, create_project_comment]
ALL_TOOLS = READ_TOOLS + CALC_TOOLS + WRITE_TOOLS
