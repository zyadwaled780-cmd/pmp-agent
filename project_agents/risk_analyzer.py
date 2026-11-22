"""Step 9 - combine the check results into a ranked risk list.

Two layers with different authority:
* DETERMINISTIC risks (``deterministic_risks``) come from numbers/rules over the checks and the update. They
  are authoritative: the LLM can neither remove nor downgrade them. The fast path and the normal path use this
  one function, so both report the same findings.
* MODEL risks are cross-cutting interpretations the LLM may add (for example a supplier delay hitting an
  already overloaded person). They are kept only when GROUNDED: they must cite evidence that appears in the
  facts, reference only known task ids and not duplicate a known risk. Ungrounded ones are dropped with a
  data-gap note. ``revise_model_risks`` is the component an evaluator failure about a model risk is routed to.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel

from checks import CheckResults
from checks.deadline_checker import DeadlineStatus
from domain.models import SEVERITY_ORDER, ExtractedUpdate, RiskItem, Severity
from project_agents.claims import normalize_text
from project_agents.llm import LLMCall, call_llm


class RiskAnalysis(BaseModel):
    risks: list[RiskItem]


INSTRUCTIONS = """You are a risk analyst for a project manager.
You receive: the latest update and the project check results (JSON).
Return ADDITIONAL risks that come from combining facts (for example a late delivery that also hits
an overloaded person). Do not repeat the listed known_risks.
Rules: every risk needs evidence quoting facts from the input; related_task_ids may only contain
ids that appear in the input; never invent facts; never make speculative inferences; if nothing new, return an empty list."""


@dataclass
class RiskOutcome:
    risks: list[RiskItem]
    gaps: list[str] = field(default_factory=list)
    model_titles: set[str] = field(default_factory=set)   # titles of the risks the LLM added


# --------------------------------------------------------------------------- deterministic layer
def _dedupe(risks: list[RiskItem]) -> list[RiskItem]:
    seen, out = set(), []
    for r in risks:
        key = r.title.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def deterministic_risks(update: ExtractedUpdate, results: CheckResults) -> list[RiskItem]:
    risks: list[RiskItem] = []

    d = results.deadline
    if d and d.status == DeadlineStatus.AT_RISK:
        risks.append(RiskItem(title="Project deadline at risk", severity=Severity.CRITICAL, evidence=d.reason, related_task_ids=[]))
    elif d and d.status == DeadlineStatus.WARNING:
        risks.append(RiskItem(title="Project deadline has little buffer", severity=Severity.MEDIUM, evidence=d.reason, related_task_ids=[]))

    if results.tasks:
        overdue_ids = {t.id for t in results.tasks.overdue}
        for t in results.tasks.blocked:
            why = t.blocker or "status is blocked"
            late = f" It is also overdue (due {t.due_date})." if t.id in overdue_ids else ""
            risks.append(RiskItem(title=f"Task blocked: {t.title}", severity=Severity.HIGH,
                                  evidence=f"Task '{t.title}' is blocked ({why}).{late}",
                                  related_task_ids=[t.id] if t.id else []))
        blocked_ids = {t.id for t in results.tasks.blocked}
        for t in results.tasks.overdue:
            if t.id in blocked_ids:
                continue                      # already reported once, as blocked + overdue
            risks.append(RiskItem(title=f"Task overdue: {t.title}", severity=Severity.MEDIUM,
                                  evidence=f"Task '{t.title}' was due {t.due_date} and is not done.",
                                  related_task_ids=[t.id] if t.id else []))
        for t in results.tasks.due_after_deadline:
            risks.append(RiskItem(title=f"Task planned after project deadline: {t.title}", severity=Severity.MEDIUM,
                                  evidence=f"Task '{t.title}' is due {t.due_date}, after the project deadline {results.tasks.deadline}.",
                                  related_task_ids=[t.id] if t.id else []))

    if results.people:
        for name in results.people.overloaded:
            days = results.people.workload_days.get(name, 0)
            risks.append(RiskItem(title=f"{name} is overloaded", severity=Severity.MEDIUM,
                                  evidence=(f"{name} has {days:g} day(s) of open work against a "
                                            f"{results.people.horizon_days:g}-day horizon."),
                                  related_task_ids=[]))
        for title in results.people.tasks_owned_by_unavailable:
            risks.append(RiskItem(title=f"Task owner unavailable: {title}", severity=Severity.HIGH,
                                  evidence=f"Task '{title}' belongs to someone who is on leave.", related_task_ids=[]))

    for b in update.blockers:
        risks.append(RiskItem(title="Blocker reported in update", severity=Severity.HIGH,
                              evidence=f"Reported by {update.sender} ({update.source.value}): {b}", related_task_ids=[]))

    # Budget: a figure in THIS update wins for this turn; otherwise the persisted (accountant) value is used.
    pct, where = update.budget_used_percent, f"{update.sender} ({update.source.value}) reports"
    if pct is None and results.budget is not None and results.budget.used_percent is not None:
        pct, where = results.budget.used_percent, "The recorded project budget shows"
    if pct is not None and pct >= 80:
        sev = Severity.CRITICAL if pct >= 100 else Severity.HIGH if pct >= 90 else Severity.MEDIUM
        risks.append(RiskItem(title="Budget nearly used up" if pct < 100 else "Budget exceeded", severity=sev,
                              evidence=f"{where} {pct:g}% of the budget used.", related_task_ids=[]))
    return _dedupe(risks)


def _rank(risks: list[RiskItem]) -> list[RiskItem]:
    return sorted(risks, key=lambda r: -SEVERITY_ORDER[r.severity])


# --------------------------------------------------------------------------- grounding of model risks
_STOP = {"that", "this", "with", "have", "from", "been", "will", "would", "which", "their", "there", "than", "into",
         "also", "only", "very", "more", "most", "such", "when", "what", "about", "because", "these", "those",
         "could", "should", "while", "where", "other", "them", "they", "your", "some", "might", "typically"}
_NUM = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[^\W\d_]+", re.U)
_ARABIC = re.compile(r"[\u0600-\u06FF]")
GROUNDING_THRESHOLD = 0.6


def fact_corpus(update: ExtractedUpdate, results: CheckResults, notes: Optional[list[str]] = None) -> str:
    return normalize_text(" ".join([update.model_dump_json(exclude_none=True),
                                    results.model_dump_json(exclude_none=True), " ".join(notes or [])]))


def _content_tokens(text: str) -> list[str]:
    toks = []
    for w in _WORD.findall(normalize_text(text)):
        if w in _STOP:
            continue
        if (_ARABIC.search(w) and len(w) >= 3) or len(w) >= 4:
            toks.append(w)
    return toks


def ungrounded_reason(risk: RiskItem, corpus: str, known_ids: set[str]) -> Optional[str]:
    """None when the risk's evidence is supported by the facts, otherwise why it is not."""
    if not risk.evidence.strip():
        return "no evidence"
    if any(i not in known_ids for i in risk.related_task_ids):
        return "references a task that does not exist"
    words = _content_tokens(risk.evidence)
    nums = {float(n) for n in _NUM.findall(normalize_text(risk.evidence))}
    corpus_nums = {float(n) for n in _NUM.findall(corpus)}
    checks = [w in corpus for w in words] + [n in corpus_nums for n in nums]
    if not checks:
        return "evidence has no checkable content"
    ratio = sum(checks) / len(checks)
    if ratio < GROUNDING_THRESHOLD:
        return f"only {ratio:.0%} of its evidence appears in the facts"
    return None


def _prompt(update: ExtractedUpdate, results: CheckResults, known_titles: list[str], notes: Optional[list[str]]) -> str:
    prompt = (
        f"update: {update.model_dump_json(exclude_none=True)}\n"
        f"checks: {results.model_dump_json(exclude_none=True)}\n"
        f"known_risks: {known_titles}"
    )
    if notes:
        prompt += f"\nworker_notes: {notes}"
    return prompt


def _accept(extra: list[RiskItem], taken_titles: set[str], corpus: str, known_ids: set[str]) -> tuple[list[RiskItem], list[str]]:
    accepted, dropped = [], []
    titles = set(taken_titles)
    for r in extra:
        key = r.title.strip().lower()
        if key in titles:
            continue
        why = ungrounded_reason(r, corpus, known_ids)
        if why:
            dropped.append(f"Model-suggested risk dropped (not grounded: {why}): {r.title[:60]}")
            continue
        accepted.append(r)
        titles.add(key)
    return accepted, dropped


# --------------------------------------------------------------------------- analysis + revision
async def analyze_risks_detailed(
    update: ExtractedUpdate,
    results: CheckResults,
    llm: Optional[LLMCall] = call_llm,
    notes: Optional[list[str]] = None,
) -> RiskOutcome:
    base = deterministic_risks(update, results)
    gaps = list(results.data_gaps)
    if llm is None:
        return RiskOutcome(_rank(base), gaps, set())

    known_ids, corpus = results.known_task_ids(), fact_corpus(update, results, notes)
    try:
        extra = await llm("Risk Analysis Agent", INSTRUCTIONS, _prompt(update, results, [r.title for r in base], notes), RiskAnalysis)
    except Exception as exc:
        gaps.append(f"LLM risk analysis unavailable, rule-based risks only ({type(exc).__name__})")
        return RiskOutcome(_rank(base), gaps, set())
    accepted, dropped = _accept(extra.risks, {r.title.strip().lower() for r in base}, corpus, known_ids)
    gaps.extend(dropped)
    return RiskOutcome(_rank(base + accepted), gaps, {r.title for r in accepted})


async def analyze_risks(
    update: ExtractedUpdate,
    results: CheckResults,
    llm: Optional[LLMCall] = call_llm,
    notes: Optional[list[str]] = None,
) -> tuple[list[RiskItem], list[str]]:
    """Returns (ranked risks, data_gaps)."""
    out = await analyze_risks_detailed(update, results, llm=llm, notes=notes)
    return out.risks, out.gaps


async def revise_model_risks(
    update: ExtractedUpdate,
    results: CheckResults,
    risks: list[RiskItem],
    model_titles: set[str],
    flagged_titles: set[str],
    feedback: str,
    llm: Optional[LLMCall],
    notes: Optional[list[str]] = None,
) -> RiskOutcome:
    """The correction path for evaluator findings about MODEL risks.

    Deterministic risks and unflagged model risks are kept untouched. The flagged model risks are re-asked
    (once) with the reviewer feedback; replacements go through the same grounding filter, so a speculative
    risk can only come back if it is now supported by evidence. Whatever stays ungrounded is removed and
    recorded as a data gap - a valid risk is never discarded to obtain a passing review.
    """
    flagged = {t for t in flagged_titles if t in model_titles}
    if not flagged:                                   # nothing this component can legitimately change
        return RiskOutcome(_rank(risks), [], set(model_titles))
    kept = [r for r in risks if r.title not in flagged]
    kept_model = {t for t in model_titles if t not in flagged}
    known_ids, corpus = results.known_task_ids(), fact_corpus(update, results, notes)
    accepted: list[RiskItem] = []
    gaps: list[str] = []
    if llm is not None and flagged:
        prompt = _prompt(update, results, [r.title for r in kept], notes) + (
            f"\nreviewer feedback about your previous extra risks: {feedback}\n"
            "Return corrected extra risks that are fully supported by the facts above (or an empty list).")
        try:
            extra = await llm("Risk Analysis Agent", INSTRUCTIONS, prompt, RiskAnalysis)
            accepted, dropped = _accept(extra.risks, {r.title.strip().lower() for r in kept}, corpus, known_ids)
            gaps.extend(dropped)
        except Exception as exc:
            gaps.append(f"Risk revision unavailable ({type(exc).__name__}); flagged model risks were removed")
    removed = flagged - {r.title for r in accepted}
    gaps.extend(f"Model-suggested risk removed after review: {t[:60]}" for t in sorted(removed))
    return RiskOutcome(_rank(kept + accepted), gaps, kept_model | {r.title for r in accepted})
