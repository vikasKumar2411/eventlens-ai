from typing import Any, Dict

from app.state.eventlens_state import EventLensState


class RecoveryStepTrackingService:
    """
    Tracks execution status for bounded recovery plan steps.

    This prevents the supervisor from repeatedly routing to the same
    current_plan_step after a recovery action has already completed.
    """

    def complete_current_step(
        self,
        state: EventLensState,
        result_summary: str = "",
    ) -> Dict[str, Any]:
        current_step = state.get("current_plan_step")

        if not current_step:
            return {
                "current_plan_step": None,
                "completed_plan_steps": state.get("completed_plan_steps", []),
            }

        completed_step = dict(current_step)
        completed_step["status"] = "completed"
        completed_step["result_summary"] = result_summary

        completed_plan_steps = list(state.get("completed_plan_steps", []))
        completed_plan_steps.append(completed_step)

        return {
            "current_plan_step": None,
            "completed_plan_steps": completed_plan_steps,
        }

    def fail_current_step(
        self,
        state: EventLensState,
        failure_reason: str = "",
    ) -> Dict[str, Any]:
        current_step = state.get("current_plan_step")

        if not current_step:
            return {
                "current_plan_step": None,
                "failed_plan_steps": state.get("failed_plan_steps", []),
            }

        failed_step = dict(current_step)
        failed_step["status"] = "failed"
        failed_step["failure_reason"] = failure_reason

        failed_plan_steps = list(state.get("failed_plan_steps", []))
        failed_plan_steps.append(failed_step)

        return {
            "current_plan_step": None,
            "failed_plan_steps": failed_plan_steps,
        }