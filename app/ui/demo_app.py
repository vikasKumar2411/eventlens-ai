import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

st.write("✅ EventLens demo_app.py is running")

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "demo_data"
    / "sample_eventlens_result.json"
)


SUGGESTED_QUESTIONS = [
    "Summarize this filing",
    "Show extracted fields",
    "What is the maturity date?",
    "Show evidence for maturity date",
    "Which fields are missing?",
    "Why did the judge fail?",
    "Show recovery plan",
    "What recovery steps did the agent run?",
    "What did the agent recover?",
    "Why did the system stop?",
    "What should a human analyst review?",
    "Is the final report reliable?",
]


def load_demo_result() -> Dict[str, Any]:
    if not DATA_PATH.exists():
        st.error(f"Demo data file not found: {DATA_PATH}")
        st.stop()

    return json.loads(DATA_PATH.read_text())


def get_judge_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return result.get("judge_result") or {}


def get_judge_status(result: Dict[str, Any]) -> str:
    judge_result = get_judge_result(result)
    return (
        judge_result.get("overall_status")
        or judge_result.get("status")
        or "unknown"
    )


def get_judge_score(result: Dict[str, Any]) -> Any:
    return get_judge_result(result).get("judge_score")


def get_summary(result: Dict[str, Any]) -> str:
    summary_result = result.get("summary_result") or {}
    return summary_result.get("summary") or "No summary generated."


def get_missing_fields(result: Dict[str, Any]) -> List[str]:
    summary_result = result.get("summary_result") or {}
    warnings = summary_result.get("warnings") or []

    missing = []

    for warning in warnings:
        if isinstance(warning, str) and warning.lower().startswith("missing "):
            field = warning.replace("Missing ", "").replace(".", "").strip()
            missing.append(field)

    return missing


def get_extracted_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    confidence_result = result.get("confidence_result") or {}
    scored_fields = confidence_result.get("scored_fields") or {}

    extracted = {}

    for field_name, payload in scored_fields.items():
        if isinstance(payload, dict):
            extracted[field_name] = {
                "value": payload.get("value"),
                "confidence": (
                    payload.get("confidence")
                    or payload.get("final_confidence")
                    or payload.get("extractor_confidence")
                ),
                "quality_guard_status": payload.get("quality_guard_status"),
                "evidence_chunk_ids": payload.get("evidence_chunk_ids"),
                "evidence_quote": payload.get("evidence_quote"),
            }

    return extracted


def answer_guided_question(question: str, result: Dict[str, Any]) -> str:
    question_lower = question.lower()

    if "summarize" in question_lower:
        return get_summary(result)

    if "extracted fields" in question_lower:
        fields = get_extracted_fields(result)

        if not fields:
            return "No extracted fields were found."

        lines = []
        for field_name, payload in fields.items():
            lines.append(
                f"- **{field_name}**: {payload.get('value')} "
                f"(confidence={payload.get('confidence')})"
            )

        return "\n".join(lines)

    if "maturity date" in question_lower and "evidence" not in question_lower:
        fields = get_extracted_fields(result)
        maturity = fields.get("maturity_date") or {}
        value = maturity.get("value")

        if value:
            return f"The maturity date appears to be **{value}**."

        return "The maturity date was not found."

    if "evidence for maturity date" in question_lower:
        fields = get_extracted_fields(result)
        maturity = fields.get("maturity_date") or {}

        evidence_quote = maturity.get("evidence_quote")
        evidence_chunk_ids = maturity.get("evidence_chunk_ids")

        if evidence_quote:
            return (
                "Evidence for maturity date:\n\n"
                f"> {evidence_quote}\n\n"
                f"Chunk IDs: `{evidence_chunk_ids}`"
            )

        return "No direct evidence quote was found for maturity_date."

    if "missing" in question_lower:
        missing = get_missing_fields(result)

        if missing:
            return "Missing or unresolved fields: **" + ", ".join(missing) + "**"

        return "No missing fields were detected from the summary warnings."

    if "judge fail" in question_lower:
        judge_status = get_judge_status(result)
        judge_score = get_judge_score(result)
        missing = get_missing_fields(result)

        return (
            f"The quality judge status is **{judge_status}** with score **{judge_score}**. "
            f"The unresolved fields are: **{', '.join(missing) if missing else 'not clearly listed'}**. "
            "The system still generated a final report, but marked it as requiring review."
        )

    if "recovery plan" in question_lower:
        recovery_plan = result.get("recovery_plan") or {}
        steps = recovery_plan.get("steps") or []

        lines = [
            f"**Goal:** {recovery_plan.get('goal')}",
            f"**Target fields:** {recovery_plan.get('target_fields')}",
            "",
            "**Steps:**",
        ]

        for step in steps:
            lines.append(
                f"{step.get('step_id')}. `{step.get('action')}` "
                f"fields={step.get('target_fields')}"
            )

        return "\n".join(lines)

    if "recovery steps" in question_lower or "agent run" in question_lower:
        completed_plan_steps = result.get("completed_plan_steps") or []

        if not completed_plan_steps:
            return "No recovery plan steps were completed."

        lines = ["**Completed recovery steps:**"]

        for step in completed_plan_steps:
            lines.append(
                f"{step.get('step_id')}. `{step.get('action')}` "
                f"fields={step.get('target_fields')} "
                f"status={step.get('status')}"
            )

        return "\n".join(lines)

    if "agent recover" in question_lower:
        preserved = result.get("preserved_recovered_fields") or {}

        if not preserved:
            return "No recovered fields were preserved."

        lines = ["**Recovered fields:**"]

        for field_name, payload in preserved.items():
            lines.append(f"- **{field_name}**: {payload.get('value')}")

        return "\n".join(lines)

    if "why did the system stop" in question_lower:
        stop_reason = result.get("plan_stop_reason")
        completed_plan_steps = result.get("completed_plan_steps") or []
        recovery_plan = result.get("recovery_plan") or {}
        planned_steps = recovery_plan.get("steps") or []

        if stop_reason:
            return f"The system stopped because: `{stop_reason}`."

        if planned_steps and len(completed_plan_steps) >= len(planned_steps):
            return (
                "The system stopped because all validated recovery plan steps completed. "
                "It generated a final report instead of looping indefinitely."
            )

        return "The system stopped after final report generation."

    if "human analyst" in question_lower or "review" in question_lower:
        missing = get_missing_fields(result)

        if missing:
            return (
                "A human analyst should review these unresolved fields: **"
                + ", ".join(missing)
                + "**."
            )

        return "A human analyst should review low-confidence fields and supporting evidence."

    if "reliable" in question_lower:
        judge_status = get_judge_status(result)
        judge_score = get_judge_score(result)
        missing = get_missing_fields(result)

        return (
            "The final report was generated, but reliability is limited. "
            f"Quality judge status: **{judge_status}**. "
            f"Judge score: **{judge_score}**. "
            f"Unresolved fields: **{', '.join(missing) if missing else 'none listed'}**."
        )

    return (
        "I can answer questions about this analyzed 8-K, extracted fields, evidence, "
        "judge result, recovery plan, completed recovery steps, and final report. "
        "Try one of the suggested questions."
    )


def render_final_report(result: Dict[str, Any]) -> None:
    final_report = result.get("final_report") or {}
    missing = get_missing_fields(result)
    judge_status = get_judge_status(result)
    judge_score = get_judge_score(result)

    st.subheader("Summary")
    st.write(get_summary(result))

    st.subheader("Quality Review")

    if judge_status in {"fail", "failed", "needs_recovery"}:
        st.warning(
            "The final report was generated, but the quality judge found unresolved fields. "
            "This report should be treated as requiring human review."
        )
        review_status = "Needs human review"
    else:
        st.success("The quality judge passed the report.")
        review_status = "Passed"

    col1, col2 = st.columns(2)
    col1.metric("Review status", review_status)
    col2.metric("Judge score", judge_score)

    st.write("**Technical judge status:**", judge_status)
    st.write("**Missing or unresolved fields:**", missing or "None listed")

    recovered = result.get("preserved_recovered_fields") or {}
    if recovered:
        st.subheader("Recovered Fields")
        for field_name, payload in recovered.items():
            st.write(f"**{field_name}:** {payload.get('value')}")

    st.subheader("Safe Stop Explanation")
    completed_plan_steps = result.get("completed_plan_steps") or []
    recovery_plan = result.get("recovery_plan") or {}
    planned_steps = recovery_plan.get("steps") or []

    if planned_steps and len(completed_plan_steps) >= len(planned_steps):
        st.info(
            "All validated recovery steps were completed. The system generated a final report "
            "instead of retrying indefinitely."
        )
    else:
        st.info("The system completed final report generation.")

    st.subheader("Final Report Metadata")
    st.write("**Status:**", final_report.get("status"))
    st.write("**Case ID:**", final_report.get("case_id"))
    st.write("**Event type:**", final_report.get("event_type"))


def render_agent_trace(result: Dict[str, Any]) -> None:
    trace = result.get("agent_trace") or []

    if not trace:
        st.info("No agent trace found.")
        return

    for step in trace:
        st.write(f"**{step.get('step_number')}. {step.get('selected_agent')}**")
        st.caption(step.get("reason"))


def render_recovery_plan(result: Dict[str, Any]) -> None:
    recovery_plan = result.get("recovery_plan") or {}
    completed_plan_steps = result.get("completed_plan_steps") or []

    st.subheader("Recovery Plan")
    st.write("**Goal:**", recovery_plan.get("goal"))
    st.write("**Target fields:**", recovery_plan.get("target_fields"))

    st.write("**Planned steps:**")
    for step in recovery_plan.get("steps", []):
        st.write(
            f"{step.get('step_id')}. `{step.get('action')}` "
            f"fields={step.get('target_fields')}"
        )

    st.divider()

    st.subheader("Completed Recovery Steps")
    for step in completed_plan_steps:
        st.write(
            f"{step.get('step_id')}. `{step.get('action')}` "
            f"fields={step.get('target_fields')} "
            f"status={step.get('status')}"
        )


def main() -> None:
    st.set_page_config(
        page_title="EventLens Agentic SEC 8-K Analyst",
        page_icon="📄",
        layout="wide",
    )

    result = load_demo_result()

    st.title("EventLens: Agentic SEC 8-K Analyst")

    st.caption(
        "Cloud demo mode using a precomputed EventLens agent run over a pre-ingested SEC 8-K filing."
    )

    st.info(
        "This public demo uses a saved analysis result. The local backend version runs the full "
        "LangGraph workflow with Qdrant retrieval, recovery planning, and judge evaluation."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Task status", result.get("task_status") or result.get("status"))

    judge_status = get_judge_status(result)
    if judge_status in {"fail", "failed", "needs_recovery"}:
        col2.metric("Quality review", "Needs review")
    else:
        col2.metric("Quality review", "Passed")

    col3.metric("Judge score", get_judge_score(result))

    tab_report, tab_trace, tab_recovery, tab_qa, tab_raw = st.tabs(
        [
            "Final Report",
            "Agent Trace",
            "Recovery Plan",
            "Ask This Filing",
            "Debug JSON",
        ]
    )

    with tab_report:
        render_final_report(result)

    with tab_trace:
        render_agent_trace(result)

    with tab_recovery:
        render_recovery_plan(result)

    with tab_qa:
        st.subheader("Ask this filing")

        st.caption(
            "Ask questions about this analyzed 8-K, extracted fields, evidence, "
            "judge result, recovery steps, or final report. This demo does not "
            "provide investment, legal, or stock prediction advice."
        )

        selected_question = st.selectbox("Suggested questions", SUGGESTED_QUESTIONS)

        custom_question = st.text_input(
            "Optional custom filing-grounded question",
            placeholder="Example: What evidence supports the maturity date?",
        )

        question = custom_question.strip() or selected_question

        if st.button("Ask"):
            st.markdown(answer_guided_question(question, result))

    with tab_raw:
        compact = {
            "case_id": result.get("case_id"),
            "event_type": result.get("event_type"),
            "task_status": result.get("task_status") or result.get("status"),
            "judge_status": get_judge_status(result),
            "judge_score": get_judge_score(result),
            "plan_stop_reason": result.get("plan_stop_reason"),
            "completed_steps": result.get("completed_steps"),
            "completed_plan_steps": [
                {
                    "step_id": step.get("step_id"),
                    "action": step.get("action"),
                    "target_fields": step.get("target_fields"),
                    "status": step.get("status"),
                }
                for step in result.get("completed_plan_steps", [])
            ],
        }

        st.json(compact)


if __name__ == "__main__":
    main()