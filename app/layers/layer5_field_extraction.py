from typing import Any, Dict

from app.services.extraction_service import ExtractionService


def extract_fields_from_evidence(
    case_id: str,
    event_type: str,
    evidence_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    service = ExtractionService()

    return service.extract_for_evidence_bundle(
        case_id=case_id,
        event_type=event_type,
        evidence_bundle=evidence_bundle,
    )


def field_extraction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for field extraction.
    """

    case_id = state.get("case_id")
    event_type = state.get("event_type")
    evidence_bundle = state.get("evidence_bundle")

    if not case_id:
        raise ValueError("Missing case_id in state")

    if not event_type:
        raise ValueError("Missing event_type in state")

    if not evidence_bundle:
        raise ValueError("Missing evidence_bundle in state")

    extraction_result = extract_fields_from_evidence(
        case_id=case_id,
        event_type=event_type,
        evidence_bundle=evidence_bundle,
    )

    completed_steps = state.get("completed_steps", [])

    if "field_extraction" not in completed_steps:
        completed_steps.append("field_extraction")

    state["extraction_result"] = extraction_result
    state["completed_steps"] = completed_steps

    return state