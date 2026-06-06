import json
from pathlib import Path

from app.graph.workflow import build_eventlens_analysis_graph


def main():
    graph = build_eventlens_analysis_graph()

    initial_state = {
        "case_id": "EVL-2024-00003",
        "event_type": "debt_or_financing",
        "goal": "Analyze this SEC 8-K filing and produce a grounded debt financing event report.",
        "next_action": None,
        "task_status": "not_started",
        "completed_steps": [],
        "agent_trace": [],
        "final_report": None,

        # Recovery state
        "recovery_attempts": 0,
        "max_recovery_attempts": 2,
        "recovery_history": [],

        # LLM fallback state
        "llm_fallback_attempted": False,
        "llm_fallback_attempts": 0,
        "max_llm_fallback_attempts": 1,
        "llm_fallback_fields": [],
        "llm_fallback_result": None,

        # Planner state
        "completed_plan_steps": [],
        "failed_plan_steps": [],
        "recovery_step_count": 0,
        "max_recovery_steps": 5,
        "plan_stop_reason": None,

        # Error state
        "errors": [],
    }

    print("Starting supervisor graph...")

    result = graph.invoke(initial_state)

    print("Graph completed.")

    print("\n==============================")
    print("FINAL STATUS")
    print("==============================")
    print("task_status:", result.get("task_status") or result.get("status"))
    print("case_id:", result.get("case_id"))
    print("event_type:", result.get("event_type"))

    print("\n==============================")
    print("AGENT TRACE")
    print("==============================")

    for step in result.get("agent_trace", []):
        print(
            f"{step.get('step_number')}. "
            f"{step.get('selected_agent')} - "
            f"{step.get('reason')}"
        )

    print("\n==============================")
    print("COMPLETED STEPS")
    print("==============================")
    print(", ".join(result.get("completed_steps", [])))

    print("\n==============================")
    print("RECOVERY PLAN")
    print("==============================")

    recovery_plan = result.get("recovery_plan") or {}
    print("goal:", recovery_plan.get("goal"))
    print("target_fields:", recovery_plan.get("target_fields"))

    for step in recovery_plan.get("steps", []):
        print(
            f"{step.get('step_id')}. "
            f"{step.get('action')} "
            f"fields={step.get('target_fields')}"
        )

    print("\n==============================")
    print("COMPLETED PLAN STEPS")
    print("==============================")

    for step in result.get("completed_plan_steps", []):
        print(
            f"{step.get('step_id')}. "
            f"{step.get('action')} "
            f"fields={step.get('target_fields')} "
            f"status={step.get('status')}"
        )

    print("\n==============================")
    print("PLAN STOP REASON")
    print("==============================")
    print(result.get("plan_stop_reason"))

    print("\n==============================")
    print("JUDGE RESULT")
    print("==============================")

    judge_result = result.get("judge_result") or {}
    print("judge_status:", judge_result.get("overall_status") or judge_result.get("status"))
    print("judge_score:", judge_result.get("judge_score"))
    print("should_recover:", judge_result.get("should_recover"))

    print("\n==============================")
    print("SUMMARY")
    print("==============================")

    summary_result = result.get("summary_result") or {}
    print(summary_result.get("summary"))

    print("\n==============================")
    print("FINAL REPORT SUMMARY")
    print("==============================")

    final_report = result.get("final_report") or {}
    print("report_status:", final_report.get("status"))
    print("report_case_id:", final_report.get("case_id"))
    print("report_event_type:", final_report.get("event_type"))

    # Save full state to file instead of printing huge payload to terminal.
    output_path = Path("outputs/latest_supervisor_graph_state.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str))

    print("\nFull graph state written to:")
    print(output_path)


if __name__ == "__main__":
    main()