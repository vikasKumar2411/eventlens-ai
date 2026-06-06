from typing import Any, Dict

from app.services.llm_recovery_supervisor_service import LLMRecoverySupervisorService


def llm_recovery_supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM Recovery Supervisor node.

    This node runs after judge failure.
    It decides which bounded recovery action should happen next:
    - llm_extraction_fallback
    - retry_retrieval
    - final_report

    It does not execute the recovery action itself.
    It only updates state with the recovery decision.
    """

    service = LLMRecoverySupervisorService()

    decision = service.decide_recovery_action(state)

    current_count = state.get("llm_recovery_decision_count", 0)
    decision_number = current_count + 1

    state["llm_recovery_decision"] = decision
    state["llm_recovery_decision_attempted"] = True
    state["llm_recovery_decision_count"] = decision_number

    state["recovery_mode"] = decision.get("next_action", "final_report")
    state["target_recovery_fields"] = decision.get("target_fields", [])
    state["failure_mode"] = decision.get("failure_mode", "unknown")

    completed_steps = list(state.get("completed_steps", []))

    if "llm_recovery_supervisor" not in completed_steps:
        completed_steps.append("llm_recovery_supervisor")

    state["completed_steps"] = completed_steps

    autonomy_decision_history = list(
        state.get("autonomy_decision_history", [])
    )

    autonomy_record = {
        "decision_number": decision_number,
        "decision_source": decision.get("decision_source", "deterministic_fallback"),
        "model": decision.get("model"),
        "selected_action": decision.get("next_action"),
        "target_fields": decision.get("target_fields", []),
        "failure_mode": decision.get("failure_mode", "unknown"),
        "reason": decision.get("reason"),
        "policy_status": "approved",
        "attempt_snapshot": {
            "llm_fallback_attempts": state.get("llm_fallback_attempts", 0),
            "max_llm_fallback_attempts": state.get("max_llm_fallback_attempts", 1),
            "recovery_attempts": state.get("recovery_attempts", 0),
            "max_recovery_attempts": state.get("max_recovery_attempts", 1),
            "llm_recovery_decision_count": decision_number,
            "max_llm_recovery_decisions": state.get("max_llm_recovery_decisions", 2),
        },
    }

    autonomy_decision_history.append(autonomy_record)
    state["autonomy_decision_history"] = autonomy_decision_history

    agent_trace = list(state.get("agent_trace", []))

    agent_trace.append(
        {
            "step_number": len(agent_trace) + 1,
            "selected_agent": "llm_recovery_supervisor",
            "reason": decision.get("reason"),
            "decision": decision,
        }
    )

    state["agent_trace"] = agent_trace

    return state