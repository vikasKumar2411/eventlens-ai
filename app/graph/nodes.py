from typing import Any, Dict
from app.state.eventlens_state import EventLensState
from app.layers.layer3_event_planning import build_event_plan
from app.layers.layer5_field_extraction import extract_fields_from_evidence
from app.layers.layer6_confidence_scoring import score_extracted_fields
from app.layers.layer7_event_summary import generate_event_summary
from app.layers.layer8_judge_evaluation import evaluate_event_analysis
from app.layers.layer9_recovery_decision import decide_recovery
from app.services.retrieval_service import RetrievalService
from app.services.supervisor_service import SupervisorService
from app.services.final_report_service import FinalReportService
from app.observability.node_tracing import traced_node
from app.services.recovery_step_tracking_service import RecoveryStepTrackingService

REQUIRED_DEBT_OR_FINANCING_FIELDS = [
    "borrower_or_issuer",
    "debt_type",
    "principal_amount",
    "maturity_date",
    "interest_rate",
    "use_of_proceeds",
    "lender_or_underwriter",
    "collateral_or_guarantee",
]


def _get_judge_status(judge_result: Dict[str, Any]) -> str:
    return (
        judge_result.get("overall_status")
        or judge_result.get("status")
        or "unknown"
    )


def _get_missing_fields_from_confidence(
    confidence_result: Dict[str, Any],
) -> list[str]:
    scored_fields = (confidence_result or {}).get("scored_fields") or {}

    missing_fields = []

    for field_name in REQUIRED_DEBT_OR_FINANCING_FIELDS:
        field = scored_fields.get(field_name) or {}
        value = field.get("value")

        if value in (None, "", "unknown", "not_found", "N/A"):
            missing_fields.append(field_name)

    return missing_fields


def _append_judge_score_history(
    state: EventLensState,
    judge_result: Dict[str, Any],
    stage: str,
) -> list[Dict[str, Any]]:
    history = list(state.get("judge_score_history", []))

    history.append(
        {
            "stage": stage,
            "judge_score": judge_result.get("judge_score"),
            "judge_status": _get_judge_status(judge_result),
            "should_recover": judge_result.get("should_recover"),
            "recovery_attempts": state.get("recovery_attempts", 0),
        }
    )

    return history


def _mark_completed(state: EventLensState, step_name: str):
    completed_steps = list(state.get("completed_steps", []))

    if step_name not in completed_steps:
        completed_steps.append(step_name)

    return completed_steps


@traced_node("supervisor")
def supervisor_node(state: EventLensState) -> EventLensState:
    supervisor = SupervisorService()
    next_action = supervisor.decide_next_action(state)

    agent_trace = list(state.get("agent_trace", []))

    agent_trace.append(
        {
            "step_number": len(agent_trace) + 1,
            "selected_agent": next_action,
            "reason": _build_supervisor_reason(state, next_action),
        }
    )

    return {
        "next_action": next_action,
        "agent_trace": agent_trace,
        "task_status": "completed" if next_action == "stop" else "running",
    }


def _build_supervisor_reason(state: EventLensState, next_action: str) -> str:
    if next_action == "event_planning":
        return "No event plan exists yet."

    if next_action == "retrieval":
        current_plan_step = state.get("current_plan_step") or {}

        if current_plan_step.get("action") == "retry_retrieval":
            return (
                "The recovery plan selected retry retrieval, so the system is rerunning "
                "retrieval to gather better evidence for weak or missing fields."
            )

        if state.get("recovery_attempts", 0) > 0:
            return (
                "Recovery cleared downstream results, so retrieval must run "
                "again with the updated plan."
            )

        return "A plan exists, but retrieval has not run yet."

    if next_action == "evidence_bundle":
        return "Retrieval results exist, but the evidence bundle has not been built yet."

    if next_action == "field_extraction":
        return "Evidence bundle exists, but fields have not been extracted yet."

    if next_action == "field_quality_guard":
        return "Fields were extracted, but quality guard validation has not run yet."

    if next_action == "confidence_scoring":
        return "Fields were extracted and validated, but confidence has not been scored yet."

    if next_action == "event_summary":
        return "Confidence was scored, but the event summary has not been generated yet."

    if next_action == "judge_evaluation":
        return "Summary exists, but judge evaluation has not run yet."

    if next_action == "recovery_planner":
        return (
            "Judge found weak or missing fields, so the system is creating a bounded "
            "multi-step recovery plan instead of choosing a single reactive recovery action."
        )

    if next_action == "recovery_plan_validator":
        return (
            "A recovery plan exists, but it must be validated against allowed actions, "
            "valid debt fields, retry budgets, duplicate attempts, and stop conditions "
            "before execution."
        )

    if next_action == "recovery_plan_executor":
        return (
            "The validated recovery plan exists, so the executor is selecting exactly "
            "one safe recovery step to run before the system re-evaluates with the judge."
        )

    if next_action == "llm_extraction_fallback":
        return (
            "The current recovery plan selected LLM extraction fallback to recover "
            "weak or missing fields."
        )

    if next_action == "recovery_decision":
        return "Judge requested recovery and recovery attempts are still available."

    if next_action == "evidence_gap_analyzer":
        return (
            "Judge still found weak or missing fields after LLM fallback, "
            "so the system is diagnosing whether the failure is due to missing evidence, "
            "weak retrieval, or extractor failure."
        )

    if next_action == "amount_recovery":
        return (
            "Evidence gap analysis found that principal_amount evidence exists, "
            "but the generic extractor failed, so the supervisor selected a specialist amount recovery agent."
        )

    if next_action == "final_report":
        return "The analysis is ready to be converted into a final report."

    return "All required work is complete."


@traced_node("event_planning")
def event_planning_node(state: EventLensState) -> EventLensState:
    case_id = state["case_id"]
    event_type = state["event_type"]

    plan = build_event_plan(case_id=case_id, event_type=event_type)

    return {
        "plan": plan,
        "completed_steps": _mark_completed(state, "event_planning"),
        "recovery_attempts": state.get("recovery_attempts", 0),
        "max_recovery_attempts": state.get("max_recovery_attempts", 1),
        "recovery_history": state.get("recovery_history", []),
    }


@traced_node("retrieval")
def retrieval_node(state: EventLensState) -> EventLensState:
    plan = state["plan"]

    retrieval_service = RetrievalService()
    retrieval_results = retrieval_service.retrieve_for_plan(plan)

    updates = {
        "retrieval_results": retrieval_results,
        "completed_steps": _mark_completed(state, "retrieval"),
    }

    current_plan_step = state.get("current_plan_step") or {}

    if current_plan_step.get("action") == "retry_retrieval":
        tracker = RecoveryStepTrackingService()

        updates.update(
            tracker.complete_current_step(
                state,
                result_summary="Retry retrieval completed."
            )
        )

    return updates


@traced_node("evidence_bundle")
def evidence_bundle_node(state: EventLensState) -> EventLensState:
    retrieval_results = state["retrieval_results"]

    evidence_bundle = {}

    for field_name, result in retrieval_results.items():
        evidence_bundle[field_name] = {
            "query": result["query"],
            "evidence_chunks": result["chunks"],
        }

    return {
        "evidence_bundle": evidence_bundle,
        "completed_steps": _mark_completed(state, "evidence_bundle"),
    }


@traced_node("field_extraction")
def field_extraction_node(state: EventLensState) -> EventLensState:
    extraction_result = extract_fields_from_evidence(
        case_id=state["case_id"],
        event_type=state["event_type"],
        evidence_bundle=state["evidence_bundle"],
    )

    preserved_recovered_fields = state.get("preserved_recovered_fields", {})

    if preserved_recovered_fields:
        extraction_result = _merge_preserved_recovered_fields(
            extraction_result=extraction_result,
            preserved_recovered_fields=preserved_recovered_fields,
        )

    return {
        "extraction_result": extraction_result,
        "completed_steps": _mark_completed(state, "field_extraction"),
    }


def _merge_preserved_recovered_fields(
    extraction_result: Dict[str, Any],
    preserved_recovered_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Re-applies high-confidence LLM fallback fields after deterministic extraction reruns.

    This prevents retry retrieval + deterministic extraction from overwriting a strong
    LLM-recovered value with None or a weaker value.

    Rule:
    - Keep preserved field if current field is missing.
    - Keep preserved field if preserved confidence is >= current confidence.
    - Otherwise keep the new deterministic field.
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


@traced_node("confidence_scoring")
def confidence_scoring_node(state: EventLensState) -> EventLensState:
    confidence_result = score_extracted_fields(
        extraction_result=state["extraction_result"],
        evidence_bundle=state["evidence_bundle"],
    )

    preserved_recovered_fields = state.get("preserved_recovered_fields", {})

    if preserved_recovered_fields:
        scored_fields = confidence_result.get("scored_fields", {})

        scored_fields = _merge_preserved_recovered_fields(
            extraction_result=scored_fields,
            preserved_recovered_fields=preserved_recovered_fields,
        )

        confidence_result["scored_fields"] = scored_fields

    return {
        "confidence_result": confidence_result,
        "completed_steps": _mark_completed(state, "confidence_scoring"),
    }


@traced_node("event_summary")
def event_summary_node(state: EventLensState) -> EventLensState:
    summary_result = generate_event_summary(
        confidence_result=state["confidence_result"],
    )

    return {
        "summary_result": summary_result,
        "completed_steps": _mark_completed(state, "event_summary"),
    }


@traced_node("judge_evaluation")
def judge_evaluation_node(state: EventLensState) -> EventLensState:
    judge_result = evaluate_event_analysis(
        plan=state["plan"],
        confidence_result=state["confidence_result"],
        summary_result=state["summary_result"],
        evidence_bundle=state["evidence_bundle"],
    )

    recovery_attempts = state.get("recovery_attempts", 0)
    llm_fallback_attempts = state.get("llm_fallback_attempts", 0)

    if recovery_attempts > 0:
        stage = "post_retrieval_recovery"
    elif llm_fallback_attempts > 0:
        stage = "post_llm_fallback"
    else:
        stage = "initial_judge"

    updates = {
        "judge_result": judge_result,
        "judge_score_history": _append_judge_score_history(
            state=state,
            judge_result=judge_result,
            stage=stage,
        ),
        "completed_steps": _mark_completed(state, "judge_evaluation"),
    }

    if recovery_attempts > 0:
        updates["post_recovery_judge_result"] = judge_result
    elif llm_fallback_attempts > 0:
        updates["post_llm_fallback_judge_result"] = judge_result

    current_plan_step = state.get("current_plan_step") or {}

    if current_plan_step.get("action") == "rerun_judge":
        tracker = RecoveryStepTrackingService()

        updates.update(
            tracker.complete_current_step(
                state,
                result_summary="Judge re-evaluation completed."
            )
        )

    return updates


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


@traced_node("recovery_decision")
def recovery_decision_node(state: EventLensState) -> EventLensState:
    current_attempts = state.get("recovery_attempts", 0)
    max_attempts = state.get("max_recovery_attempts", 1)

    pre_recovery_confidence_result = state.get("confidence_result") or {}
    pre_recovery_missing_fields = _get_missing_fields_from_confidence(
        pre_recovery_confidence_result
    )

    recovery_result = decide_recovery(
        plan=state["plan"],
        judge_result=state["judge_result"],
        confidence_result=state["confidence_result"],
        recovery_attempts=current_attempts,
        max_recovery_attempts=max_attempts,
    )

    recovery_history = list(state.get("recovery_history", []))

    recovery_history.append(
        {
            "attempt_number": current_attempts + 1,
            "judge_status": state["judge_result"].get("overall_status")
            or state["judge_result"].get("status"),
            "judge_score": state["judge_result"].get("judge_score"),
            "should_recover": recovery_result.get("should_recover"),
            "fields_to_recover": recovery_result.get("fields_to_recover", []),
            "recovery_reason": recovery_result.get("recovery_reason"),
            "missing_fields_before_recovery": pre_recovery_missing_fields,
        }
    )

    updates = {
        "recovery_result": recovery_result,
        "recovery_attempts": current_attempts + 1,
        "recovery_history": recovery_history,
        "completed_steps": _mark_completed(state, "recovery_decision"),
    }

    if recovery_result.get("should_recover"):
        if not state.get("pre_recovery_judge_result"):
            updates["pre_recovery_judge_result"] = state.get("judge_result") or {}

        updates["plan"] = recovery_result.get("updated_plan", state["plan"])

        updates["field_improvement_history"] = list(
            state.get("field_improvement_history", [])
        ) + [
            {
                "stage": "pre_recovery",
                "recovery_attempts": current_attempts + 1,
                "missing_fields_before_recovery": pre_recovery_missing_fields,
                "fields_to_recover": recovery_result.get("fields_to_recover", []),
            }
        ]

        # Clear downstream state so supervisor reruns retrieval onward.
        updates["retrieval_results"] = {}
        updates["evidence_bundle"] = {}
        updates["extraction_result"] = {}
        updates["confidence_result"] = {}
        updates["summary_result"] = {}
        updates["judge_result"] = {}

        # Remove downstream completed steps so the graph can rerun them.
        completed_steps = [
            step for step in updates["completed_steps"]
            if step
            not in {
                "retrieval",
                "evidence_bundle",
                "field_extraction",
                "field_quality_guard",
                "confidence_scoring",
                "event_summary",
                "judge_evaluation",
            }
        ]

        updates["completed_steps"] = completed_steps

    return updates


@traced_node("final_report")
def final_report_node(state: EventLensState) -> EventLensState:
    service = FinalReportService()
    final_report = service.build_report(state)

    return {
        "final_report": final_report,
        "completed_steps": _mark_completed(state, "final_report"),
        "task_status": "completed",
        "next_action": "stop",
    }