from app.state.eventlens_state import EventLensState
from app.services.amount_recovery_service import AmountRecoveryService
from app.observability.node_tracing import traced_node
from app.services.recovery_step_tracking_service import RecoveryStepTrackingService
from app.services.recovery_step_tracking_service import RecoveryStepTrackingService

@traced_node("amount_recovery")
def amount_recovery_node(state: EventLensState) -> EventLensState:
    service = AmountRecoveryService()

    amount_recovery_result = service.recover(
        evidence_bundle=state.get("evidence_bundle") or {},
        confidence_result=state.get("confidence_result") or {},
    )

    updates = {
        "amount_recovery_result": amount_recovery_result,
        "completed_steps": _mark_completed(state, "amount_recovery"),
    }

    if amount_recovery_result.get("recovered"):
        recovered_field = amount_recovery_result.get("recovered_field") or {}

        confidence_result = dict(state.get("confidence_result") or {})
        scored_fields = dict(confidence_result.get("scored_fields") or {})

        scored_fields["principal_amount"] = recovered_field
        confidence_result["scored_fields"] = scored_fields

        preserved_recovered_fields = dict(
            state.get("preserved_recovered_fields") or {}
        )
        preserved_recovered_fields["principal_amount"] = recovered_field

        updates["confidence_result"] = confidence_result
        updates["preserved_recovered_fields"] = preserved_recovered_fields


    tracker = RecoveryStepTrackingService()

    updates.update(
        tracker.complete_current_step(
            state,
            result_summary="Amount recovery specialist completed."
        )
    )

    return updates


def _mark_completed(state: EventLensState, step_name: str):
    completed_steps = list(state.get("completed_steps", []))

    if step_name not in completed_steps:
        completed_steps.append(step_name)

    return completed_steps