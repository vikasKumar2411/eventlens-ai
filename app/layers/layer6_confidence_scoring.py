from typing import Any, Dict

from app.services.confidence_service import ConfidenceService


def score_extracted_fields(
    extraction_result: Dict[str, Any],
    evidence_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    service = ConfidenceService()

    return service.score_extraction_result(
        extraction_result=extraction_result,
        evidence_bundle=evidence_bundle,
    )


def confidence_scoring_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for confidence scoring.
    """

    extraction_result = state.get("extraction_result")
    evidence_bundle = state.get("evidence_bundle")

    if not extraction_result:
        raise ValueError("Missing extraction_result in state")

    if not evidence_bundle:
        raise ValueError("Missing evidence_bundle in state")

    confidence_result = score_extracted_fields(
        extraction_result=extraction_result,
        evidence_bundle=evidence_bundle,
    )

    completed_steps = state.get("completed_steps", [])

    if "confidence_scoring" not in completed_steps:
        completed_steps.append("confidence_scoring")

    state["confidence_result"] = confidence_result
    state["completed_steps"] = completed_steps

    return state