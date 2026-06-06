from typing import Any, Dict

from app.services.summary_service import SummaryService


def generate_event_summary(
    confidence_result: Dict[str, Any],
) -> Dict[str, Any]:
    service = SummaryService()

    return service.generate_summary(
        confidence_result=confidence_result,
    )


def event_summary_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for event summary generation.
    """

    confidence_result = state.get("confidence_result")

    if not confidence_result:
        raise ValueError("Missing confidence_result in state")

    summary_result = generate_event_summary(
        confidence_result=confidence_result,
    )

    completed_steps = state.get("completed_steps", [])

    if "event_summary" not in completed_steps:
        completed_steps.append("event_summary")

    state["summary_result"] = summary_result
    state["completed_steps"] = completed_steps

    return state