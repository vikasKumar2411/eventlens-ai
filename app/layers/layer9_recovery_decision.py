from typing import Any, Dict

from app.services.recovery_service import RecoveryService


def decide_recovery(
    plan: Dict[str, Any],
    judge_result: Dict[str, Any],
    confidence_result: Dict[str, Any],
    recovery_attempts: int,
    max_recovery_attempts: int,
) -> Dict[str, Any]:
    service = RecoveryService()

    return service.decide_and_rewrite(
        plan=plan,
        judge_result=judge_result,
        confidence_result=confidence_result,
        recovery_attempts=recovery_attempts,
        max_recovery_attempts=max_recovery_attempts,
    )


def recovery_decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for recovery decision.
    """

    plan = state.get("plan")
    judge_result = state.get("judge_result")
    confidence_result = state.get("confidence_result")

    recovery_attempts = state.get("recovery_attempts", 0)
    max_recovery_attempts = state.get("max_recovery_attempts", 1)

    if not plan:
        raise ValueError("Missing plan in state")

    if not judge_result:
        raise ValueError("Missing judge_result in state")

    if not confidence_result:
        raise ValueError("Missing confidence_result in state")

    recovery_result = decide_recovery(
        plan=plan,
        judge_result=judge_result,
        confidence_result=confidence_result,
        recovery_attempts=recovery_attempts,
        max_recovery_attempts=max_recovery_attempts,
    )

    recovery_history = state.get("recovery_history", [])
    recovery_history.append(recovery_result)

    completed_steps = state.get("completed_steps", [])

    if "recovery_decision" not in completed_steps:
        completed_steps.append("recovery_decision")

    state["recovery_result"] = recovery_result
    state["recovery_history"] = recovery_history
    state["recovery_attempts"] = recovery_attempts + 1
    state["completed_steps"] = completed_steps

    return state