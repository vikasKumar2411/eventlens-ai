from langgraph.graph import StateGraph, START, END

from app.state.eventlens_state import EventLensState
from app.graph.nodes import (
    supervisor_node,
    event_planning_node,
    retrieval_node,
    evidence_bundle_node,
    field_extraction_node,
    confidence_scoring_node,
    event_summary_node,
    judge_evaluation_node,
    recovery_decision_node,
    final_report_node,
)

from app.layers.layer5b_field_quality_guard import field_quality_guard_node
from app.layers.layer5c_llm_extraction_fallback import llm_extraction_fallback_node
from app.layers.layer8b_llm_recovery_supervisor import llm_recovery_supervisor_node
from app.layers.layer8c_evidence_gap_analyzer import evidence_gap_analyzer_node
from app.layers.layer8d_amount_recovery import amount_recovery_node

# New bounded planning nodes
from app.layers.layer8e_recovery_planner import recovery_planner_node
from app.layers.layer8f_plan_validator import recovery_plan_validator_node
from app.layers.layer8g_plan_executor import recovery_plan_executor_node


def supervisor_router(state: EventLensState) -> str:
    return state.get("next_action", "stop")


def build_eventlens_analysis_graph():
    workflow = StateGraph(EventLensState)

    # Core deterministic pipeline
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("event_planning", event_planning_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("evidence_bundle", evidence_bundle_node)
    workflow.add_node("field_extraction", field_extraction_node)
    workflow.add_node("field_quality_guard", field_quality_guard_node)
    workflow.add_node("confidence_scoring", confidence_scoring_node)
    workflow.add_node("event_summary", event_summary_node)
    workflow.add_node("judge_evaluation", judge_evaluation_node)

    # Existing recovery nodes
    workflow.add_node("llm_recovery_supervisor", llm_recovery_supervisor_node)
    workflow.add_node("llm_extraction_fallback", llm_extraction_fallback_node)
    workflow.add_node("evidence_gap_analyzer", evidence_gap_analyzer_node)
    workflow.add_node("amount_recovery", amount_recovery_node)
    workflow.add_node("recovery_decision", recovery_decision_node)

    # New bounded planning loop
    workflow.add_node("recovery_planner", recovery_planner_node)
    workflow.add_node("recovery_plan_validator", recovery_plan_validator_node)
    workflow.add_node("recovery_plan_executor", recovery_plan_executor_node)

    # Final output
    workflow.add_node("final_report", final_report_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            # Core deterministic pipeline
            "event_planning": "event_planning",
            "retrieval": "retrieval",
            "evidence_bundle": "evidence_bundle",
            "field_extraction": "field_extraction",
            "field_quality_guard": "field_quality_guard",
            "confidence_scoring": "confidence_scoring",
            "event_summary": "event_summary",
            "judge_evaluation": "judge_evaluation",

            # Existing recovery nodes
            "llm_recovery_supervisor": "llm_recovery_supervisor",
            "llm_extraction_fallback": "llm_extraction_fallback",
            "evidence_gap_analyzer": "evidence_gap_analyzer",
            "amount_recovery": "amount_recovery",
            "recovery_decision": "recovery_decision",

            # New bounded planning nodes
            "recovery_planner": "recovery_planner",
            "recovery_plan_validator": "recovery_plan_validator",
            "recovery_plan_executor": "recovery_plan_executor",

            # Finalization
            "final_report": "final_report",
            "stop": END,
        },
    )

    # Everything returns to supervisor because supervisor owns routing.
    workflow.add_edge("event_planning", "supervisor")
    workflow.add_edge("retrieval", "supervisor")
    workflow.add_edge("evidence_bundle", "supervisor")
    workflow.add_edge("field_extraction", "supervisor")
    workflow.add_edge("field_quality_guard", "supervisor")
    workflow.add_edge("confidence_scoring", "supervisor")
    workflow.add_edge("event_summary", "supervisor")
    workflow.add_edge("judge_evaluation", "supervisor")

    workflow.add_edge("llm_recovery_supervisor", "supervisor")
    workflow.add_edge("llm_extraction_fallback", "supervisor")
    workflow.add_edge("evidence_gap_analyzer", "supervisor")
    workflow.add_edge("amount_recovery", "supervisor")
    workflow.add_edge("recovery_decision", "supervisor")

    workflow.add_edge("recovery_planner", "supervisor")
    workflow.add_edge("recovery_plan_validator", "supervisor")
    workflow.add_edge("recovery_plan_executor", "supervisor")

    workflow.add_edge("final_report", END)

    return workflow.compile()