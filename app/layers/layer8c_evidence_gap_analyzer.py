from app.state.eventlens_state import EventLensState
from app.services.evidence_gap_analyzer_service import EvidenceGapAnalyzerService
from app.observability.node_tracing import traced_node
from app.services.recovery_step_tracking_service import RecoveryStepTrackingService

@traced_node("evidence_gap_analyzer")
def evidence_gap_analyzer_node(state: EventLensState) -> EventLensState:
    service = EvidenceGapAnalyzerService()

    target_fields = (
        state.get("target_recovery_fields")
        or _get_missing_or_weak_fields(state)
    )

    evidence_gap_analysis = service.analyze(
        confidence_result=state.get("confidence_result") or {},
        evidence_bundle=state.get("evidence_bundle") or {},
        target_fields=target_fields,
    )

    tracker = RecoveryStepTrackingService()

    updates = {
        "evidence_gap_analysis": evidence_gap_analysis,
        "completed_steps": _mark_completed(state, "evidence_gap_analyzer"),
    }

    updates.update(
        tracker.complete_current_step(
            state,
            result_summary="Evidence gap analysis completed."
        )
    )

    return updates


def _get_missing_or_weak_fields(state: EventLensState) -> list[str]:
    confidence_result = state.get("confidence_result") or {}
    scored_fields = confidence_result.get("scored_fields") or {}

    fields = []

    for field_name, field in scored_fields.items():
        value = field.get("value")
        final_confidence = field.get("final_confidence") or 0.0

        if value in (None, "", "unknown", "not_found", "N/A"):
            fields.append(field_name)
            continue

        if final_confidence < 0.6:
            fields.append(field_name)

    return fields


def _mark_completed(state: EventLensState, step_name: str):
    completed_steps = list(state.get("completed_steps", []))

    if step_name not in completed_steps:
        completed_steps.append(step_name)

    return completed_steps