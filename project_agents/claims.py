"""Detection of unsupported FACTUAL claims in generated text (English + Egyptian/Modern Standard Arabic).

Two kinds are detected:
* completion claims  - "everything is done", "100% complete", "كل المهام خلصت", "المشروع اكتمل" ...
* delivery promises  - "we will deliver on Friday", "we guarantee", "هنسلم يوم الخميس", "نضمن التسليم" ...

Recommendations are NOT claims: "recommend asking the supplier for a firm date", "يُنصح بتسليم الوثائق قبل
الخميس" describe what a person should do and are allowed. Only first-person commitments and statements
that something IS finished are flagged. A claim that is negated right before it ("not all tasks are done",
"لم يكتمل المشروع") is not a claim either.

This is one signal among several: semantic action checks live in evaluator_agent.grounding_findings and the
LLM evaluator still reviews every report. Regexes alone are never the only guard.
"""
import re

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٪", "01234567890123456789%")
_AR_MARKS = re.compile("[\u064b-\u065f\u0670\u0640]")          # tashkeel + tatweel
_AR_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})


def normalize_text(text: str) -> str:
    """Lowercase, Arabic-Indic digits -> ASCII, strip diacritics, unify alef/yaa/taa-marbuta variants."""
    t = (text or "").translate(_AR_DIGITS)
    t = _AR_MARKS.sub("", t).translate(_AR_LETTERS)
    return t.lower()


_DONE_AR = r"(?:خلص\w*|انتهي\w*|انتهت\b|اكتمل\w*|مكتمل\w*|منتهي\w*|منجز\w*|تمت\b|تم\b)"

COMPLETION_PATTERNS = [re.compile(p, re.I) for p in [
    # English
    r"\b100\s*%\s*(?:done|complete|completed|finished)",
    r"\b(?:done|complete|completed|finished)\s*(?:at|by)?\s*100\s*%",
    r"\bfully\s+(?:complete|completed|done|finished)\b",
    r"\ball\s+(?:the\s+)?(?:tasks|work|deliverables|items)\s+(?:are|is|have been)\s+(?:done|complete|completed|finished)\b",
    r"\beverything\s+(?:is|has been)\s+(?:done|complete|completed|finished)\b",
    r"\bthe\s+project\s+is\s+(?:now\s+)?(?:fully\s+)?(?:done|complete|completed|finished|delivered)\b",
    # Arabic (text is normalized first)
    rf"(?:كل|جميع)\s+(?:ال)?(?:مهام|اعمال|شغل|مراحل|مخرجات|حاجه|اشياء|شيء)\s+{_DONE_AR}",
    rf"(?:ال)?مشروع\s+(?:كله\s+|بالكامل\s+)?{_DONE_AR}",
    r"تم\s+(?:ال)?(?:انتهاء|انجاز|اكمال)\s+من\s+(?:كل|جميع)",
    r"(?:خلصنا|انجزنا|انهينا|اكملنا)\s+(?:كل|جميع)",
    r"100\s*%\s*(?:مكتمل\w*|خلصان|خلص\w*|منجز\w*)",
    r"(?:مكتمل\w*|خلصان|منجز\w*)\s*(?:بنسبه\s*)?100\s*%",
]]

PROMISE_PATTERNS = [re.compile(p, re.I) for p in [
    # English first-person commitments and absolute guarantees
    r"\b(?:we|i)\s*(?:will|'ll|shall)\s+(?:deliver|ship|finish|complete)\b",
    r"\bwe\s+(?:guarantee|promise|commit\s+to)\b",
    r"\bi\s+(?:guarantee|promise|assure)\b",
    r"\bguarantee[sd]?\b",
    r"\bpromise[sd]?\b",
    r"\b(?:will\s+be|is\s+guaranteed\s+to\s+be)\s+(?:delivered|done|completed|finished)\s+(?:on|by)\b",
    # Arabic
    r"(?:هنسلم|هنخلص|هنكمل|هنجهز|هننتهي|سنسلم|سنكمل|سننتهي|سنجهز)\w*\s+(?:\S+\s+){0,2}(?:يوم|في|قبل|بحلول|الاسبوع|بكره|بكرا|غدا|خلال)",
    r"\b(?:نضمن|اضمن|هنضمن|هضمن|نتعهد|اتعهد|اعدكم|اوعدكم)\b",
    r"(?:التسليم|الانتهاء|الموعد)\s+مضمون",
]]

_NEGATION = re.compile(r"\b(?:not|isn't|aren't|never|cannot|can't|no|without|neither)\b|(?:^|\s)(?:مش|ليس|ليست|لم|لن|لا)\s")


def _claimed(patterns, text: str) -> bool:
    norm = normalize_text(text)
    for p in patterns:
        for m in p.finditer(norm):
            before = norm[max(0, m.start() - 25):m.start()]
            if _NEGATION.search(before + " "):
                continue                    # "not all tasks are done", "لم يكتمل المشروع"
            return True
    return False


def has_completion_claim(text: str) -> bool:
    return _claimed(COMPLETION_PATTERNS, text)


def has_delivery_promise(text: str) -> bool:
    return _claimed(PROMISE_PATTERNS, text)
