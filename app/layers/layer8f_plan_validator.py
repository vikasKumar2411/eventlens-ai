from typing import Any, Dict

from app.services.recovery_plan_validator_service import RecoveryPlanValidatorService
from app.observability.node_tracing import traced_node


validator_service = RecoveryPlanValidatorService()


def _mark_completed(state: Dict[str, Any], step_name: str) -> list[str]:
    completed_steps = list(state.get("completed_steps", []))

    if step_name not in completed_steps:
        completed_steps.append(step_name)

    return completed_steps


@traced_node("recovery_plan_validator")
def recovery_plan_validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    validated_plan = validator_service.validate_plan(state)

    updates = {
        "validated_recovery_plan": validated_plan,
        "completed_steps": _mark_completed(state, "recovery_plan_validator"),
    }

    if not validated_plan.get("is_valid"):
        updates["plan_stop_reason"] = "no_valid_steps_remaining"

    return updates