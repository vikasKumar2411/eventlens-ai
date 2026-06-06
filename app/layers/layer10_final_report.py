from typing import Any, Dict

from app.services.final_report_service import FinalReportService


def final_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates the final structured EventLens report.
    """

    service = FinalReportService()
    final_report = service.build_report(state)

    completed_steps = state.get("completed_steps", [])

    if "final_report" not in completed_steps:
        completed_steps.append("final_report")

    state["final_report"] = final_report
    state["completed_steps"] = completed_steps
    state["task_status"] = "completed"
    state["next_action"] = "stop"

    return state