from typing import Any, Dict, List, Optional


class RecoveryPlanExecutorService:
    """
    Selects one validated step to execute.
    It does not run business logic directly.
    The graph routes to the appropriate specialist node.
    """

    def get_next_step(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        validated_plan = state.get("validated_recovery_plan", {}) or {}
        steps = validated_plan.get("steps", []) or []

        completed = state.get("completed_plan_steps", []) or []
        failed = state.get("failed_plan_steps", []) or []

        for step in steps:
            if not self._already_seen(step, completed, failed):
                return step

        return None

    def _already_seen(
        self,
        step: Dict[str, Any],
        completed: List[Dict[str, Any]],
        failed: List[Dict[str, Any]],
    ) -> bool:
        all_seen = completed + failed

        step_action = step.get("action")
        step_fields = step.get("target_fields") or []

        return any(
            seen_step.get("action") == step_action
            and (seen_step.get("target_fields") or []) == step_fields
            for seen_step in all_seen
        )