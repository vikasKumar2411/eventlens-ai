from typing import Any, Dict

from app.services.supervisor_service import SupervisorService


def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor node that chooses the next agent.
    """

    supervisor = SupervisorService()
    next_action = supervisor.decide_next_action(state)

    agent_trace = state.get("agent_trace", [])

    agent_trace.append(
        {
            "step_number": len(agent_trace) + 1,
            "selected_agent": next_action,
            "reason": _build_reason(state, next_action),
        }
    )

    state["next_action"] = next_action
    state["agent_trace"] = agent_trace
    state["task_status"] = "completed" if next_action == "stop" else "running"

    return state


def _build_reason(state: Dict[str, Any], next_action: str) -> str:
    reasons = {
        "event_planning": "No plan exists yet.",
        "retrieval": "A plan exists, but no evidence bundle exists yet.",
        "field_extraction": "Evidence exists, but fields have not been extracted yet.",
        "confidence_scoring": "Fields were extracted, but confidence has not been scored yet.",
        "event_summary": "Confidence was scored, but the event summary has not been generated yet.",
        "judge_evaluation": "A summary exists, but the judge has not evaluated it yet.",
        "recovery_decision": "The judge found issues and recovery is still allowed.",
        "final_report": "The analysis is ready to be converted into a final report.",
        "stop": "All required work is complete.",
    }

    return reasons.get(next_action, "Supervisor selected the next best action.")