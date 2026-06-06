from typing import Any, Dict

from app.services.retrieval_service import RetrievalService


def retrieve_evidence_for_plan(
    case_id: str,
    event_type: str,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    service = RetrievalService()

    return service.retrieve_for_plan(
        case_id=case_id,
        event_type=event_type,
        plan=plan,
    )


def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for retrieval.

    Expected output:
        state["retrieval_results"]
        state["evidence_bundle"]
    """

    case_id = state.get("case_id")
    event_type = state.get("event_type")
    plan = state.get("plan")

    if not case_id:
        raise ValueError("Missing case_id in state")

    if not event_type:
        raise ValueError("Missing event_type in state")

    if not plan:
        raise ValueError("Missing plan in state")

    retrieval_results = retrieve_evidence_for_plan(
        case_id=case_id,
        event_type=event_type,
        plan=plan,
    )

    completed_steps = state.get("completed_steps", [])

    if "retrieval" not in completed_steps:
        completed_steps.append("retrieval")

    state["retrieval_results"] = retrieval_results

    if "evidence_bundle" in retrieval_results:
        state["evidence_bundle"] = retrieval_results["evidence_bundle"]
    else:
        state["evidence_bundle"] = retrieval_results

    state["completed_steps"] = completed_steps

    return state