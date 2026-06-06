from typing import Any, Dict, List

from app.services.llm_extraction_fallback_service import LLMExtractionFallbackService
from app.services.recovery_step_tracking_service import RecoveryStepTrackingService

def llm_extraction_fallback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for LLM extraction fallback.

    Runs only after judge failure and only for weak/missing fields.

    Important:
    - Updates extraction_result with LLM recovered fields.
    - Stores high-confidence LLM recovered fields separately in
      preserved_recovered_fields so later retry retrieval does not erase them.
    """

    case_id = state.get("case_id")
    event_type = state.get("event_type")
    extraction_result = state.get("extraction_result")
    confidence_result = state.get("confidence_result")
    evidence_bundle = state.get("evidence_bundle")
    judge_result = state.get("judge_result")

    if not case_id:
        raise ValueError("Missing case_id in state")

    if not event_type:
        raise ValueError("Missing event_type in state")

    if not extraction_result:
        raise ValueError("Missing extraction_result in state")

    if not confidence_result:
        raise ValueError("Missing confidence_result in state")

    if not evidence_bundle:
        raise ValueError("Missing evidence_bundle in state")

    if not judge_result:
        raise ValueError("Missing judge_result in state")

    service = LLMExtractionFallbackService()

    fallback_result = service.run_fallback(
        case_id=case_id,
        event_type=event_type,
        extraction_result=extraction_result,
        confidence_result=confidence_result,
        evidence_bundle=evidence_bundle,
        judge_result=judge_result,
    )

    updated_extraction_result = fallback_result["updated_extraction_result"]

    preserved_recovered_fields = _build_preserved_recovered_fields(
        existing_preserved_fields=state.get("preserved_recovered_fields", {}),
        updated_extraction_result=updated_extraction_result,
        fallback_result=fallback_result,
        target_fields=fallback_result.get("target_fields", []),
    )

    completed_steps = list(state.get("completed_steps", []))

    if "llm_extraction_fallback" not in completed_steps:
        completed_steps.append("llm_extraction_fallback")

    llm_fallback_attempts = state.get("llm_fallback_attempts", 0) + 1

    tracker = RecoveryStepTrackingService()

    updates = {
        "extraction_result": updated_extraction_result,
        "preserved_recovered_fields": preserved_recovered_fields,
        "llm_fallback_result": fallback_result,
        "llm_fallback_attempted": True,
        "llm_fallback_attempts": llm_fallback_attempts,
        "llm_fallback_fields": fallback_result.get("target_fields", []),
        "completed_steps": completed_steps,

        # Clear downstream results so the supervisor reruns validation/scoring/judging.
        "confidence_result": {},
        "summary_result": {},
        "judge_result": {},
        "final_report": None,
    }

    updates.update(
        tracker.complete_current_step(
            state,
            result_summary="LLM extraction fallback completed."
        )
    )

    return updates


def _build_preserved_recovered_fields(
    existing_preserved_fields: Dict[str, Any],
    updated_extraction_result: Dict[str, Any],
    fallback_result: Dict[str, Any],
    target_fields: List[str],
) -> Dict[str, Any]:
    """
    Stores high-confidence LLM fallback fields so they can survive later
    retrieval/extraction retries.

    This function is intentionally defensive because the fallback service may store
    LLM field outputs in different shapes:
    - updated_extraction_result[field]
    - fallback_result["field_results"][field]
    - fallback_result["llm_field_results"][field]
    - fallback_result["results"][field]
    """

    preserved = dict(existing_preserved_fields or {})

    for field_name in target_fields:
        normalized_field = _get_best_llm_field(
            field_name=field_name,
            updated_extraction_result=updated_extraction_result,
            fallback_result=fallback_result,
        )

        if not normalized_field:
            continue

        value = normalized_field.get("value")
        final_confidence = normalized_field.get("final_confidence") or 0.0
        extractor_confidence = normalized_field.get("extractor_confidence") or 0.0

        has_value = value not in (None, "", "N/A")
        is_high_confidence = final_confidence >= 0.85 or extractor_confidence >= 0.85

        if has_value and is_high_confidence:
            preserved[field_name] = normalized_field

    return preserved


def _get_best_llm_field(
    field_name: str,
    updated_extraction_result: Dict[str, Any],
    fallback_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Finds and normalizes the LLM fallback result for one field.
    """

    updated_field = updated_extraction_result.get(field_name)

    if isinstance(updated_field, dict):
        value = updated_field.get("value")
        method = updated_field.get("extraction_method")
        final_confidence = updated_field.get("final_confidence") or 0.0
        extractor_confidence = updated_field.get("extractor_confidence") or 0.0

        if (
            value not in (None, "", "N/A")
            and method == "llm_fallback"
            and (final_confidence >= 0.85 or extractor_confidence >= 0.85)
        ):
            return updated_field

    candidate_maps = [
        fallback_result.get("field_results", {}),
        fallback_result.get("llm_field_results", {}),
        fallback_result.get("results", {}),
        fallback_result.get("fields", {}),
    ]

    for candidate_map in candidate_maps:
        if not isinstance(candidate_map, dict):
            continue

        raw_field = candidate_map.get(field_name)

        if not isinstance(raw_field, dict):
            continue

        normalized = _normalize_llm_field_result(
            field_name=field_name,
            raw_field=raw_field,
            updated_field=updated_field,
        )

        value = normalized.get("value")
        final_confidence = normalized.get("final_confidence") or 0.0
        extractor_confidence = normalized.get("extractor_confidence") or 0.0

        if value not in (None, "", "N/A") and (
            final_confidence >= 0.85 or extractor_confidence >= 0.85
        ):
            return normalized

    return {}


def _normalize_llm_field_result(
    field_name: str,
    raw_field: Dict[str, Any],
    updated_field: Any,
) -> Dict[str, Any]:
    """
    Converts the LLM fallback field result into the same shape as extraction_result.
    """

    value = raw_field.get("value")

    confidence = (
        raw_field.get("final_confidence")
        or raw_field.get("extractor_confidence")
        or raw_field.get("confidence")
        or 0.0
    )

    evidence_quote = (
        raw_field.get("evidence_quote")
        or raw_field.get("quote")
        or raw_field.get("supporting_quote")
    )

    reason = raw_field.get("reason") or raw_field.get("rationale") or ""

    evidence_chunk_ids = []

    if isinstance(updated_field, dict):
        evidence_chunk_ids = updated_field.get("evidence_chunk_ids", [])

    return {
        "value": value,
        "extraction_method": "llm_fallback",
        "extractor_confidence": confidence,
        "evidence_quality_score": 0.9 if confidence >= 0.85 else 0.5,
        "final_confidence": min(0.96, confidence),
        "quality_guard_status": "pass" if confidence >= 0.85 else "review",
        "reason": reason or "Recovered by LLM fallback from supporting evidence.",
        "evidence_quote": evidence_quote,
        "evidence_chunk_ids": evidence_chunk_ids,
    }