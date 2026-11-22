"""
Guardrails Module for PMP Agent System
Implements Input Guardrail, Data Safety Scan, and Output Guardrail.
"""

import re
from typing import List, Tuple

from .models import InputMessage, SecurityAssessment


class InputGuardrail:
    """
    Step 2 in PMP Process:
    Examines incoming natural language text for prompt injections,
    jailbreaks, and manipulative instructions.
    """

    SUSPICIOUS_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous\s+)?(rules|instructions|prompts)",
        r"(?i)تجاهل\s+(كل\s+)?(القواعد|التعليمات|الأوامر|السابقة)",
        r"(?i)disregard\s+(the\s+)?rules",
        r"(?i)system\s*:\s*override",
        r"(?i)اعتبر\s+(كل\s+)?(المهام|المشاريع)\s+(خلصت|مكتملة|تمت)",
        r"(?i)mark\s+all\s+tasks?\s+(as\s+)?(done|completed|finished)",
        r"(?i)احذف\s+كل\s+مهام",
        r"(?i)delete\s+all\s+(tasks?|projects?|records?)",
        r"(?i)drop\s+table",
        r"(?i)you\s+are\s+now\s+an\s+unrestricted\s+ai",
        r"(?i)انت\s+الآن\s+بدون\s+قيود",
    ]

    @classmethod
    def scan_message(cls, message: InputMessage) -> SecurityAssessment:
        text = message.text.strip()
        for pattern in cls.SUSPICIOUS_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return SecurityAssessment(
                    is_safe=False,
                    threat_type="PROMPT_INJECTION",
                    reason=f"Detected suspicious command pattern: '{match.group(0)}'",
                    quarantined=True,
                    details={"matched_pattern": pattern, "source": message.source}
                )

        return SecurityAssessment(
            is_safe=True,
            threat_type=None,
            reason="Input passed security check",
            quarantined=False
        )


class DataSafetyGuardrail:
    """
    Step 8 in PMP Process:
    Scans data retrieved from external systems, tools, and comments
    for indirect prompt injection attempts.
    """

    INDIRECT_INJECTION_PATTERNS = [
        r"(?i)assign\s+all\s+tasks?\s+to\s+",
        r"(?i)اسند\s+كل\s+المهام\s+لـ",
        r"(?i)change\s+admin\s+to",
        r"(?i)give\s+access\s+to",
        r"(?i)<script[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
    ]

    @classmethod
    def sanitize_external_data(cls, raw_data: str) -> Tuple[str, bool, List[str]]:
        cleaned_data = raw_data
        warnings = []
        has_violations = False

        for pattern in cls.INDIRECT_INJECTION_PATTERNS:
            matches = list(re.finditer(pattern, cleaned_data))
            if matches:
                has_violations = True
                for m in matches:
                    snippet = m.group(0)
                    warnings.append(f"Sanitized indirect injection payload: '{snippet}'")
                    cleaned_data = re.sub(pattern, "[REDACTED_SUSPICIOUS_PAYLOAD]", cleaned_data)

        return cleaned_data, has_violations, warnings


class OutputGuardrail:
    """
    Step 13 in PMP Process:
    Validates report output to ensure no unverified date commitments
    or unfounded claims of 100% completion without cryptographic/system confirmation.
    """

    UNVERIFIED_PROMISE_PATTERNS = [
        r"(?i)100%\s*(done|finished|مكتمل|خلصان)",
        r"(?i)أضمن\s+(لك\s+)?(التسليم|الانتهاء|الموعد)",
        r"(?i)guarantee\s+(completion|delivery)\s+by",
        r"(?i)لا\s+يوجد\s+أي\s+خطر\s+نهائياً",
        r"(?i)zero\s+risk\s+whatsoever",
    ]

    @classmethod
    def validate_output(cls, report_text: str, is_system_verified: bool = False) -> Tuple[bool, List[str]]:
        issues = []
        if not is_system_verified:
            for pattern in cls.UNVERIFIED_PROMISE_PATTERNS:
                matches = re.findall(pattern, report_text)
                if matches:
                    issues.append(f"Unverified absolute claim detected: '{matches[0]}'")

        return len(issues) == 0, issues
