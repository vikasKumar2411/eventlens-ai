from typing import Any, Dict

from app.services.recovery_planner_service import RecoveryPlannerService
from app.observability.node_tracing import traced_node


planner_service = RecoveryPlannerService()


def _mark_completed(state: Dict[str, Any], step_name: str) -> list[str]:
    completed_steps = list(state.get("completed_steps", []))

    if step_name not in completed_steps:
        completed_steps.append(step_name)

    return completed_steps


@traced_node("recovery_planner")
def recovery_planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    recovery_plan = planner_service.create_plan(state)

    return {
        "recovery_goal": recovery_plan.get("goal"),
        "recovery_plan": recovery_plan,

        # Clear stale validation/execution state whenever a fresh plan is created.
        "validated_recovery_plan": {},
        "current_plan_step": None,
        "plan_stop_reason": None,

        # Preserve plan execution history.
        "completed_plan_steps": state.get("completed_plan_steps", []),
        "failed_plan_steps": state.get("failed_plan_steps", []),

        # Preserve bounded execution budget.
        "recovery_step_count": state.get("recovery_step_count", 0),
        "max_recovery_steps": state.get("max_recovery_steps", 5),

        "completed_steps": _mark_completed(state, "recovery_planner"),
    }