"""OFFLINE DEMO ONLY: a rule-based stand-in that has the same signature as call_llm.

It is NOT a language model. It lets the UI, the tests and a demo run without an API key.
Real runs must use project_agents.llm.call_llm.
"""
import ast
import json
import re

from domain.models import ExtractedUpdate, ProposedAction, SourceType, TaskItem
from project_agents.evaluator_agent import EvalResult
from project_agents.orchestrator import OrchestrationPlan
from project_agents.risk_analyzer import RiskAnalysis
from project_agents.status_writer import ReportDraft

BLOCK_WORDS = re.compile(r"\b(late|delay\w*|down|blocked|stuck|waiting|missing|outage)\b|تأخر|تتأخر|واقف|معطل|مش شغال|متأخر", re.I)
DONE_WORDS = re.compile(r"\b(finished|done|completed|complete)\b|خلصت|خلصنا|انتهيت|انتهى", re.I)
WORKLOAD_WORDS = re.compile(r"overload|busy|workload|who is|مشغول|ضغط", re.I)


def make_offline_llm(known_tasks: list[tuple[str, str]]):
    """known_tasks: [(task_id, title)] so the stand-in can recognise task names in a message."""

    async def llm(name: str, instructions: str, prompt: str, output_type):
        if output_type is ExtractedUpdate:
            return _extract(prompt, known_tasks)
        if output_type is OrchestrationPlan:
            if WORKLOAD_WORDS.search(prompt):
                return OrchestrationPlan(workers=["people", "tasks"], focus="workload and late tasks")
            return OrchestrationPlan(workers=["tasks", "people", "deadline"], focus="full project status")
        if output_type is RiskAnalysis:
            return RiskAnalysis(risks=[])
        if output_type is ReportDraft:
            return _write(prompt)
        if output_type is EvalResult:
            return EvalResult(passed=True, issues=[], feedback="ok")
        raise TypeError(f"offline llm cannot produce {output_type}")

    return llm


def _extract(prompt: str, known_tasks) -> ExtractedUpdate:
    m = re.search(r"Sender: (.*?) \(role: (\w+)\)", prompt)
    sender, role = (m.group(1), m.group(2)) if m else ("unknown", "employee")
    body = re.search(r"<message>\n(.*?)\n</message>", prompt, re.S)
    text = body.group(1).strip() if body else prompt
    blocked = bool(BLOCK_WORDS.search(text))
    done = bool(DONE_WORDS.search(text))

    tasks = []
    for tid, title in known_tasks:
        if title.lower() in text.lower():
            status = "done" if done else "blocked" if blocked else "in_progress"
            dm = re.search(r"(\d+)\s*days?", text, re.I)
            tasks.append(TaskItem(id=tid, title=title, owner=None, status=status, due_date=None,
                                  estimate_days=None, blocker=None,
                                  delay_days=int(dm.group(1)) if (dm and blocked) else None))
    pct = None
    pm = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if pm and re.search(r"budget|spen[dt]|ميزانية|مصروف", text, re.I):
        pct = float(pm.group(1))
    return ExtractedUpdate(
        source=SourceType(role), sender=sender, summary=text[:140], tasks=tasks,
        blockers=[text[:140]] if blocked else [], risks=[],
        mentioned_deadline=None, budget_used_percent=pct,
    )


def _write(prompt: str) -> ReportDraft:
    upd_line = re.search(r"^update: (.*)$", prompt, re.M)
    risk_line = re.search(r"^risks: (.*)$", prompt, re.M)
    update = json.loads(upd_line.group(1)) if upd_line else {}
    risks = ast.literal_eval(risk_line.group(1)) if risk_line else []
    sender = update.get("sender", "there")
    tasks = update.get("tasks", [])
    blockers = update.get("blockers", [])
    has_update = bool(tasks or blockers or update.get("budget_used_percent") is not None)

    if risks:
        top = risks[0]
        risk_summary = f"{len(risks)} risk(s) found. Top risk: {top['title']} ({top['severity']}). {top['evidence']}"
        evidence = top["evidence"]
    else:
        risk_summary, evidence = "No major risks found in current checks.", "Checks indicate standard operation."

    if not has_update:
        summary = (
            f"### Direct Reply\nHello {sender}! Here is the current executive status of the project.\n\n"
            f"### Critical Risks & Blockers\n"
            f"- **T2 (Payment API)** is blocked waiting for supplier sandbox keys.\n"
            f"- **T4 (Documentation)** is overdue.\n"
            f"- **Deadline & Workload**: Zero schedule buffer; Sara is overloaded with multiple open items.\n\n"
            f"### Recommended Next Steps\n"
            f"- Follow up urgently with the Payment API supplier for sandbox keys.\n"
            f"- Reassign Documentation (T4) or rebalance Sara's load to restore buffer."
        )
        actions = []
    else:
        summary = (
            f"### Direct Reply\nProcessed project update from {sender}.\n\n"
            f"### Critical Risks & Blockers\n{risk_summary}\n\n"
            f"### Recommended Next Steps\nReview the recorded findings and monitor open critical path items."
        )
        actions = [ProposedAction(
            action="create_project_comment", comment=f"Status check: {risk_summary}",
            reason="Keep the project record up to date.", evidence=evidence
        )]
        for t in tasks:
            if t.get("status") == "done" and t.get("id"):
                actions.append(ProposedAction(
                    action="update_task", task_id=t["id"], status="done", title=t.get("title"),
                    reason="The update says this task is finished.",
                    evidence=f"Reported by {update.get('sender')} ({update.get('source')})."
                ))
    return ReportDraft(summary=summary, proposed_actions=actions)

