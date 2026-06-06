from typing import Any, Dict

from app.services.recovery_plan_executor_service import RecoveryPlanExecutorService
from app.observability.node_tracing import traced_node


executor_service = RecoveryPlanExecutorService()


def _mark_completed(state: Dict[str, Any], step_name: str) -> list[str]:
    completed_steps = list(state.get("completed_steps", []))

    if step_name not in completed_steps:
        completed_steps.append(step_name)

    return completed_steps


@traced_node("recovery_plan_executor")
def recovery_plan_executor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    next_step = executor_service.get_next_step(state)
    completed_steps = _mark_completed(state, "recovery_plan_executor")

    if not next_step:
        return {
            "current_plan_step": None,
            "plan_stop_reason": "no_valid_steps_remaining",
            "completed_steps": completed_steps,
        }

    if next_step.get("action") == "retry_retrieval":
        completed_steps = [
            step for step in completed_steps
            if step not in {
                "retrieval",
                "evidence_bundle",
                "field_extraction",
                "field_quality_guard",
                "confidence_scoring",
                "event_summary",
                "judge_evaluation",
            }
        ]

        return {
            "current_plan_step": next_step,
            "recovery_step_count": state.get("recovery_step_count", 0) + 1,

            # Clear downstream state so supervisor reruns retrieval onward.
            "retrieval_results": {},
            "evidence_bundle": {},
            "extraction_result": {},
            "confidence_result": {},
            "summary_result": {},
            "judge_result": {},

            "completed_steps": completed_steps,
        }

    return {
        "current_plan_step": next_step,
        "recovery_step_count": state.get("recovery_step_count", 0) + 1,
        "completed_steps": completed_steps,
    }