"""
Action Gate Module for PMP Agent System
Step 14 in PMP Process: Deterministic policy engine
"""

from typing import List, Tuple

from .models import ActionPolicyVerdict, ActionType, ProposedAction, RouteType


class ActionGate:
    @classmethod
    def evaluate_action(cls, action: ProposedAction, route: RouteType = RouteType.NORMAL) -> Tuple[ActionPolicyVerdict, str]:
        if action.action_type == ActionType.DELETE_TASK:
            return (
                ActionPolicyVerdict.FORBIDDEN,
                "Deleting tasks is strictly prohibited by system governance policy."
            )

        if route == RouteType.CRITICAL_PATH:
            if action.action_type != ActionType.ADD_COMMENT:
                return (
                    ActionPolicyVerdict.NEEDS_APPROVAL,
                    "Project is on CRITICAL PATH. All mutations require human manager authorization."
                )

        if action.action_type in [ActionType.UPDATE_DUE_DATE, ActionType.REASSIGN_TASK]:
            return (
                ActionPolicyVerdict.NEEDS_APPROVAL,
                f"Modifying task schedules or reassigning owners ({action.action_type.value}) requires manager confirmation."
            )

        if action.action_type == ActionType.ADD_COMMENT:
            return (
                ActionPolicyVerdict.ALLOWED,
                "Audit comments and informational notes are authorized for automatic execution."
            )

        if action.action_type == ActionType.CREATE_TASK:
            if action.payload.get("priority") == "CRITICAL":
                return (
                    ActionPolicyVerdict.NEEDS_APPROVAL,
                    "Creating a CRITICAL priority task requires manager review."
                )
            return (
                ActionPolicyVerdict.ALLOWED,
                "Routine task creation is pre-authorized."
            )

        return (
            ActionPolicyVerdict.NEEDS_APPROVAL,
            "Action type requires manual managerial confirmation."
        )

    @classmethod
    def process_actions(cls, actions: List[ProposedAction], route: RouteType = RouteType.NORMAL) -> List[ProposedAction]:
        for act in actions:
            verdict, reason = cls.evaluate_action(act, route=route)
            act.verdict = verdict
            act.verdict_reason = reason
        return actions
