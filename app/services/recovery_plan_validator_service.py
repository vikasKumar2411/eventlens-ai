from typing import Any, Dict, List


ALLOWED_ACTIONS = {
    "llm_extraction_fallback",
    "evidence_gap_analysis",
    "amount_recovery",
    "retry_retrieval",
    "rerun_judge",
}


VALID_DEBT_FIELDS = {
    "borrower_or_issuer",
    "debt_type",
    "principal_amount",
    "maturity_date",
    "interest_rate",
    "use_of_proceeds",
    "lender_or_underwriter",
    "collateral_or_guarantee",
}


class RecoveryPlanValidatorService:
    """
    Validates recovery plans before execution.

    This keeps EventLens agentic but bounded:
    - Planner can propose recovery steps.
    - Validator decides whether those steps are safe and allowed.
    - Executor only receives validated steps.
    """

    def validate_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        recovery_plan = state.get("recovery_plan") or {}
        steps = recovery_plan.get("steps") or []

        validated_steps = []
        rejected_steps = []

        max_steps = state.get("max_recovery_steps", 5)

        for step in steps[:max_steps]:
            validation = self.validate_step(step=step, state=state)

            if validation["is_valid"]:
                validated_steps.append(step)
            else:
                rejected_step = dict(step)
                rejected_step["rejection_reason"] = validation["reason"]
                rejected_steps.append(rejected_step)

        return {
            "goal": recovery_plan.get("goal"),
            "target_fields": recovery_plan.get("target_fields", []),
            "steps": validated_steps,
            "rejected_steps": rejected_steps,
            "stop_conditions": recovery_plan.get("stop_conditions", []),
            "is_valid": len(validated_steps) > 0,
            "validator_type": "deterministic_plan_validator",
        }

    def validate_step(
        self,
        step: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        action = step.get("action")
        target_fields = step.get("target_fields") or []

        if action not in ALLOWED_ACTIONS:
            return {
                "is_valid": False,
                "reason": f"Action not allowed: {action}",
            }

        if not isinstance(target_fields, list):
            return {
                "is_valid": False,
                "reason": "target_fields must be a list.",
            }

        invalid_fields = [
            field for field in target_fields
            if field not in VALID_DEBT_FIELDS
        ]

        if invalid_fields:
            return {
                "is_valid": False,
                "reason": f"Invalid target fields: {invalid_fields}",
            }

        if state.get("recovery_step_count", 0) >= state.get("max_recovery_steps", 5):
            return {
                "is_valid": False,
                "reason": "Max recovery steps reached.",
            }

        completed = state.get("completed_plan_steps") or []
        failed = state.get("failed_plan_steps") or []

        if self._step_already_seen(step=step, seen_steps=completed):
            return {
                "is_valid": False,
                "reason": "Step already completed.",
            }

        if self._step_already_seen(step=step, seen_steps=failed):
            return {
                "is_valid": False,
                "reason": "Step already failed.",
            }

        return {
            "is_valid": True,
            "reason": "Step is valid.",
        }

    def _step_already_seen(
        self,
        step: Dict[str, Any],
        seen_steps: List[Dict[str, Any]],
    ) -> bool:
        step_action = step.get("action")
        step_fields = step.get("target_fields") or []

        return any(
            seen_step.get("action") == step_action
            and (seen_step.get("target_fields") or []) == step_fields
            for seen_step in seen_steps
        )