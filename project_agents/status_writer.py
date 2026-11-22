"""Step 11 - write the management summary and propose actions.

The writer owns exactly two fields of the report: ``summary`` and ``proposed_actions``. Risks belong to the
Risk Analyzer; revisions requested by the evaluator reach the writer only for the fields it can change.
Filtering here (forbidden action kinds) is a convenience: the authoritative validation is
tools.validation.validate_action, applied again by the action gate and the executor.
"""
from typing import Optional

from pydantic import BaseModel

from checks import CheckResults
from domain.models import ExecutiveReport, ExtractedUpdate, ProposedAction, RiskItem
from project_agents.llm import GuardrailTripped, LLMCall, call_llm

# delete_task is never proposed; the action gate also forbids it (defence in depth).
ALLOWED_ACTIONS = {"create_task", "update_task", "assign_task", "create_project_comment"}


class ReportDraft(BaseModel):
    summary: str
    proposed_actions: list[ProposedAction]


INSTRUCTIONS = """You write a short status report for a project manager as an AI Project Operations Assistant, in {language}.
Use ONLY the facts in the input JSON. Do not invent any tasks, people, numbers, or dates.

Structure the summary with these exact markdown headers (keep the header text in English, write the body in {language}):
### Direct Reply
A warm, professional reply to the sender's message. If it was a greeting or general inquiry, acknowledge it and summarize the overall project status in 1-2 clear sentences.

### Critical Risks & Blockers
Concise bullet points with the key blockers and schedule/team risks that appear in the checks and risks input.

### Recommended Next Steps
Concrete recommendations for the project manager. Recommendations say what SHOULD be done; they are not statements that something is done or promised.

Rules:
- Never say a task is done/finished unless its status is "done" in the input.
- Never say or imply that everything/the project is complete when other tasks remain open (in any language).
- Never promise a delivery date or outcome ("we will deliver on ...", "guaranteed", "هنسلم يوم ...", "نضمن ...").
- Never include meta-commentary, debugging notes, or text discussing reviewer feedback. Write ONLY the final report.
- proposed_actions: only create_task, update_task, assign_task, create_project_comment. Never delete anything.
  * Propose update_task only when the incoming update reports a real change compared to the current state in the input.
  * Never propose an update, assignment or task that repeats the current state (same status, same due date, same owner).
  * assign_task/create_task owners must be available team members from the input (not on leave).
  * Every action needs a reason and evidence that quotes the input. If no action is justified, return an empty list.
- If reviewer feedback is given, fix exactly those issues."""


async def write_report(
    update: ExtractedUpdate,
    risks: list[RiskItem],
    results: CheckResults,
    data_gaps: list[str],
    feedback: Optional[str] = None,
    previous: Optional[ExecutiveReport] = None,
    llm: LLMCall = call_llm,
    language: str = "English",
    notes: Optional[list[str]] = None,
    allow_actions: bool = True,
    model_risk_titles: Optional[set[str]] = None,
) -> ExecutiveReport:
    prompt = (
        f"update: {update.model_dump_json(exclude_none=True)}\n"
        f"checks: {results.model_dump_json(exclude_none=True)}\n"
        f"risks: {[r.model_dump(mode='json') for r in risks]}\n"
        f"data_gaps: {data_gaps}"
    )
    if notes:
        prompt += f"\nworker_notes (hints only, the checks above are the source of truth): {notes}"
    if feedback and previous:
        prompt += (f"\n\nIMPORTANT: the reviewer found these problems in your previous draft: {feedback}\n"
                   "Write a corrected final report fixing exactly these problems. Do NOT mention the review or feedback.")

    try:
        draft = await llm("Status Writer", INSTRUCTIONS.format(language=language), prompt, ReportDraft)
    except GuardrailTripped as exc:  # the SDK output guardrail rejected the draft: one more try
        retry = prompt + f"\n\nYour last draft was blocked by a guardrail: {exc.info}. Remove promises and unproven claims."
        draft = await llm("Status Writer", INSTRUCTIONS.format(language=language), retry, ReportDraft)
    if isinstance(draft, dict):
        draft = ReportDraft.model_validate(draft)

    gaps = list(data_gaps)
    kept = [a for a in draft.proposed_actions if a.action in ALLOWED_ACTIONS]
    if len(kept) != len(draft.proposed_actions):
        gaps.append("Writer proposed a forbidden action; it was removed.")
    if not allow_actions:                    # e.g. a greeting / status check: never mutate the project
        kept = []
    return ExecutiveReport(summary=draft.summary, risks=risks, proposed_actions=kept, data_gaps=gaps,
                           model_risk_titles=sorted(model_risk_titles or ()))
