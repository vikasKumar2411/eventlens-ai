from typing import Any, Dict

from app.services.field_quality_guard_service import FieldQualityGuardService


def field_quality_guard_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node that validates extracted fields before confidence scoring.

    Important:
    - Validates extraction_result normally.
    - Re-applies high-confidence preserved LLM fallback fields after validation
      so the guard does not overwrite a previously recovered strong value.
    """

    extraction_result = state.get("extraction_result")
    evidence_bundle = state.get("evidence_bundle")

    if not extraction_result:
        raise ValueError("Missing extraction_result in state")

    if not evidence_bundle:
        raise ValueError("Missing evidence_bundle in state")

    service = FieldQualityGuardService()

    guarded_extraction_result = service.validate_extraction_result(
        extraction_result=extraction_result,
        evidence_bundle=evidence_bundle,
    )

    preserved_recovered_fields = state.get("preserved_recovered_fields", {})

    if preserved_recovered_fields:
        guarded_extraction_result = _merge_preserved_recovered_fields(
            extraction_result=guarded_extraction_result,
            preserved_recovered_fields=preserved_recovered_fields,
        )

    completed_steps = list(state.get("completed_steps", []))

    if "field_quality_guard" not in completed_steps:
        completed_steps.append("field_quality_guard")

    return {
        "extraction_result": guarded_extraction_result,
        "completed_steps": completed_steps,
    }


def _merge_preserved_recovered_fields(
    extraction_result: Dict[str, Any],
    preserved_recovered_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Re-applies high-confidence LLM fallback fields after quality guard validation.

    Rule:
    - Keep preserved field if current field is missing.
    - Keep preserved field if preserved confidence is >= current confidence.
    """

    merged_result = dict(extraction_result or {})

    for field_name, preserved_field in preserved_recovered_fields.items():
        if not isinstance(preserved_field, dict):
            continue

        preserved_value = preserved_field.get("value")
        preserved_confidence = (
            preserved_field.get("final_confidence")
            or preserved_field.get("extractor_confidence")
            or 0.0
        )

        if preserved_value in (None, "", "N/A"):
            continue

        current_field = merged_result.get(field_name)

        if not isinstance(current_field, dict):
            merged_result[field_name] = preserved_field
            continue

        current_value = current_field.get("value")
        current_confidence = (
            current_field.get("final_confidence")
            or current_field.get("extractor_confidence")
            or 0.0
        )

        current_missing = current_value in (None, "", "N/A")
        preserved_is_better_or_equal = preserved_confidence >= current_confidence

        if current_missing or preserved_is_better_or_equal:
            merged_result[field_name] = preserved_field

    return merged_result