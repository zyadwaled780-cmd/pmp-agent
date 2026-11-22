"""
Deterministic Router for PMP Agent System
Step 5 in PMP Process: Fixed code rules to choose between Normal Path and Critical Path.
"""

import re

from .models import ExtractedUpdate, RouteDecision, RouteType


class DeterministicRouter:
    CRITICAL_BLOCKER_KEYWORDS = [
        r"(?i)blocking\s+(the\s+)?(release|deployment|launch|production)",
        r"(?i)system\s+is\s+down",
        r"(?i)عائق\s+حرج",
        r"(?i)معطل\s+(التسليم|الإطلاق|الإنتاج|النظام)",
        r"(?i)توقف\s+(النظام|السيرفر|الدفع)",
        r"(?i)critical\s+blocker",
    ]

    @classmethod
    def evaluate_route(cls, update: ExtractedUpdate) -> RouteDecision:
        for task in update.tasks:
            if task.is_blocker:
                return RouteDecision(
                    route=RouteType.CRITICAL_PATH,
                    requires_immediate_alert=True,
                    reason=f"Task '{task.title}' is flagged as a critical blocker."
                )

            if task.delay_days >= 3:
                return RouteDecision(
                    route=RouteType.CRITICAL_PATH,
                    requires_immediate_alert=True,
                    reason=f"Task '{task.title}' introduces a significant delay of {task.delay_days} days."
                )

        raw_text = str(update.raw_entities.get("raw_text", ""))
        for pattern in cls.CRITICAL_BLOCKER_KEYWORDS:
            match = re.search(pattern, raw_text)
            if match:
                return RouteDecision(
                    route=RouteType.CRITICAL_PATH,
                    requires_immediate_alert=True,
                    reason=f"Critical blocker detected in update text: '{match.group(0)}'"
                )

        return RouteDecision(
            route=RouteType.NORMAL,
            requires_immediate_alert=False,
            reason="Update contains standard operational progress or low-impact notices."
        )
