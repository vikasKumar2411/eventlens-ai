from typing import Any, Dict

from app.services.judge_service import JudgeService


def evaluate_event_analysis(
    plan: Dict[str, Any],
    confidence_result: Dict[str, Any],
    summary_result: Dict[str, Any],
    evidence_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    service = JudgeService()

    return service.evaluate(
        plan=plan,
        confidence_result=confidence_result,
        summary_result=summary_result,
        evidence_bundle=evidence_bundle,
    )


def judge_evaluation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for judge evaluation.
    """

    plan = state.get("plan")
    confidence_result = state.get("confidence_result")
    summary_result = state.get("summary_result")
    evidence_bundle = state.get("evidence_bundle")

    if not plan:
        raise ValueError("Missing plan in state")

    if not confidence_result:
        raise ValueError("Missing confidence_result in state")

    if not summary_result:
        raise ValueError("Missing summary_result in state")

    if not evidence_bundle:
        raise ValueError("Missing evidence_bundle in state")

    judge_result = evaluate_event_analysis(
        plan=plan,
        confidence_result=confidence_result,
        summary_result=summary_result,
        evidence_bundle=evidence_bundle,
    )

    completed_steps = state.get("completed_steps", [])

    if "judge_evaluation" not in completed_steps:
        completed_steps.append("judge_evaluation")

    state["judge_result"] = judge_result
    state["completed_steps"] = completed_steps

    return state