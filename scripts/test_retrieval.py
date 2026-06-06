import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from app.graph.workflow import build_eventlens_analysis_graph
from app.observability.tracing import setup_tracing, get_tracer


setup_tracing()


def build_initial_state(case_id: str, event_type: str) -> dict:
    return {
        "case_id": case_id,
        "event_type": event_type,
        "goal": "Analyze this SEC 8-K filing and produce a grounded debt financing event report.",
        "next_action": None,
        "task_status": "not_started",
        "completed_steps": [],
        "agent_trace": [],
        "final_report": None,

        # Recovery state
        "recovery_attempts": 0,
        "max_recovery_attempts": 1,
        "recovery_history": [],

        # LLM fallback state
        "llm_fallback_attempted": False,
        "llm_fallback_attempts": 0,
        "max_llm_fallback_attempts": 1,
        "llm_fallback_fields": [],
        "llm_fallback_result": None,

        # LLM recovery supervisor state
        "llm_recovery_decision": {},
        "llm_recovery_decision_attempted": False,
        "llm_recovery_decision_count": 0,
        "max_llm_recovery_decisions": 2,
        "recovery_mode": "",
        "target_recovery_fields": [],
        "failure_mode": "",
        "autonomy_decision_history": [],

        # Preserved high-confidence recovered fields
        "preserved_recovered_fields": {},

        # Measurable recovery state
        "pre_recovery_judge_result": {},
        "post_llm_fallback_judge_result": {},
        "post_recovery_judge_result": {},
        "judge_score_history": [],
        "field_improvement_history": [],

        # Evidence gap analysis state
        "evidence_gap_analysis": {},

        # Error state
        "errors": [],
    }


def invoke_graph_with_trace(
    graph,
    initial_state: dict,
    case_id: str,
    event_type: str,
) -> dict:
    tracer = get_tracer()

    with tracer.start_as_current_span("eventlens.run") as span:
        span.set_attribute("eventlens.case_id", case_id)
        span.set_attribute("eventlens.event_type", event_type)
        span.set_attribute("eventlens.entrypoint", "scripts.test_retrieval")

        final_state = graph.invoke(initial_state)

        judge_result = final_state.get("judge_result") or {}

        span.set_attribute(
            "eventlens.judge_status",
            judge_result.get("overall_status")
            or judge_result.get("status")
            or "unknown",
        )
        span.set_attribute(
            "eventlens.judge_score",
            float(judge_result.get("judge_score") or 0.0),
        )
        span.set_attribute(
            "eventlens.should_recover",
            bool(judge_result.get("should_recover") or False),
        )
        span.set_attribute(
            "eventlens.recovery_attempts",
            int(final_state.get("recovery_attempts", 0) or 0),
        )
        span.set_attribute(
            "eventlens.llm_fallback_attempts",
            int(final_state.get("llm_fallback_attempts", 0) or 0),
        )

        completed_steps = final_state.get("completed_steps") or []
        span.set_attribute("eventlens.completed_steps", ",".join(completed_steps))

        llm_fields = final_state.get("llm_fallback_fields") or []
        span.set_attribute("eventlens.llm_fallback_fields", ",".join(llm_fields))

        return final_state


def print_agent_trace(final_state: dict) -> None:
    print("\n" + "=" * 80)
    print("Agent Trace")
    print("=" * 80)

    agent_trace = final_state.get("agent_trace", [])

    if not agent_trace:
        print("No agent trace found.")
        return

    for step in agent_trace:
        print(
            f"{step.get('step_number')}. Supervisor chose "
            f"{step.get('selected_agent')} because {step.get('reason')}"
        )


def print_autonomy_decision_history(final_state: dict) -> None:
    print("\n" + "=" * 80)
    print("Autonomy Decision History")
    print("=" * 80)

    final_report = final_state.get("final_report") or {}

    history = (
        final_report.get("autonomy_decision_history")
        or final_state.get("autonomy_decision_history")
        or []
    )

    if not history:
        print("No autonomy decision history found.")
        return

    for decision in history:
        print("\n" + "-" * 60)
        print(f"Decision #: {decision.get('decision_number')}")
        print(f"Source: {decision.get('decision_source')}")
        print(f"Model: {decision.get('model')}")
        print(f"Selected Action: {decision.get('selected_action')}")
        print(f"Target Fields: {decision.get('target_fields')}")
        print(f"Failure Mode: {decision.get('failure_mode')}")
        print(f"Policy Status: {decision.get('policy_status')}")
        print(f"Reason: {decision.get('reason')}")


def print_amount_recovery(final_state: dict) -> None:
    print("\n" + "=" * 80)
    print("Amount Recovery")
    print("=" * 80)

    result = final_state.get("amount_recovery_result") or {}

    if not result:
        print("No amount recovery result found.")
        return

    print(f"Recovered: {result.get('recovered')}")
    print(f"Reason: {result.get('reason')}")

    recovered_field = result.get("recovered_field") or {}

    if recovered_field:
        print(f"Recovered Value: {recovered_field.get('value')}")
        print(f"Final Confidence: {recovered_field.get('final_confidence')}")
        print(f"Evidence Quote: {recovered_field.get('evidence_quote')}")

    candidates = result.get("candidates") or []

    if candidates:
        print("\nTop Candidates:")
        for candidate in candidates[:5]:
            print(
                f"- amount={candidate.get('amount')}, "
                f"confidence={candidate.get('confidence')}, "
                f"context_hits={candidate.get('context_hits')}, "
                f"chunk_id={candidate.get('chunk_id')}"
            )


def print_summary(final_state: dict) -> None:
    print("\n" + "=" * 80)
    print("Event Summary")
    print("=" * 80)

    summary_result = final_state.get("summary_result", {})

    print(f"Summary: {summary_result.get('summary')}")
    print(f"Summary Confidence: {summary_result.get('summary_confidence')}")
    print(f"Summary Method: {summary_result.get('summary_method')}")

    warnings = summary_result.get("warnings") or []
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


def print_judge_result(final_state: dict) -> None:
    print("\n" + "=" * 80)
    print("Judge Result")
    print("=" * 80)

    judge_result = final_state.get("judge_result", {})

    if not judge_result:
        print("No judge result found.")
        return

    print(f"Overall Status: {judge_result.get('overall_status')}")
    print(f"Judge Score: {judge_result.get('judge_score')}")
    print(f"Should Recover: {judge_result.get('should_recover')}")

    failed_checks = judge_result.get("failed_checks") or []

    if failed_checks:
        print("\nFailed Checks:")
        for check in failed_checks:
            print(f"- {check.get('name')}: {check.get('message')}")


def print_recovery_metrics(final_state: dict) -> None:
    print("\n" + "=" * 80)
    print("Recovery Metrics")
    print("=" * 80)

    pre = final_state.get("pre_recovery_judge_result") or {}
    post_llm = final_state.get("post_llm_fallback_judge_result") or {}
    post = final_state.get("post_recovery_judge_result") or {}

    history = final_state.get("judge_score_history") or []
    field_history = final_state.get("field_improvement_history") or []

    print(f"Pre-recovery judge score: {pre.get('judge_score')}")
    print(
        "Pre-recovery status: "
        f"{pre.get('overall_status') or pre.get('status')}"
    )

    print(f"Post-LLM fallback judge score: {post_llm.get('judge_score')}")
    print(
        "Post-LLM fallback status: "
        f"{post_llm.get('overall_status') or post_llm.get('status')}"
    )

    print(f"Post-recovery judge score: {post.get('judge_score')}")
    print(
        "Post-recovery status: "
        f"{post.get('overall_status') or post.get('status')}"
    )

    if pre.get("judge_score") is not None and post.get("judge_score") is not None:
        improvement = round(post.get("judge_score") - pre.get("judge_score"), 4)
        print(f"Retrieval recovery score improvement: {improvement}")

    print("\nJudge score history:")
    if not history:
        print("No judge score history found.")
    else:
        for item in history:
            print(
                f"- {item.get('stage')}: "
                f"score={item.get('judge_score')}, "
                f"status={item.get('judge_status')}, "
                f"recovery_attempts={item.get('recovery_attempts')}"
            )

    print("\nField improvement history:")
    if not field_history:
        print("No field improvement history found.")
    else:
        for item in field_history:
            print(
                f"- stage={item.get('stage')}, "
                f"attempt={item.get('recovery_attempts')}, "
                f"missing_before={item.get('missing_fields_before_recovery')}, "
                f"fields_to_recover={item.get('fields_to_recover')}, "
                f"recovered={item.get('recovered_fields')}, "
                f"still_missing={item.get('still_missing_fields')}"
            )


def print_evidence_gap_analysis(final_state: dict) -> None:
    print("\n" + "=" * 80)
    print("Evidence Gap Analysis")
    print("=" * 80)

    analysis = final_state.get("evidence_gap_analysis") or {}

    if not analysis:
        print("No evidence gap analysis found.")
        return

    field_gap_analysis = analysis.get("field_gap_analysis") or {}

    if not field_gap_analysis:
        print("No field-level evidence gap analysis found.")
        return

    for field_name, field_result in field_gap_analysis.items():
        print("\n" + "-" * 60)
        print(f"Field: {field_name}")
        print(f"Failure Type: {field_result.get('failure_type')}")
        print(f"Reason: {field_result.get('reason')}")
        print(f"Current Value: {field_result.get('current_value')}")
        print(f"Final Confidence: {field_result.get('final_confidence')}")
        print(f"Extractor Confidence: {field_result.get('extractor_confidence')}")
        print(f"Evidence Quality Score: {field_result.get('evidence_quality_score')}")
        print(f"Keyword Hits: {field_result.get('keyword_hits')}")
        print(f"Evidence Chunk Count: {field_result.get('evidence_chunk_count')}")

        samples = field_result.get("sample_evidence") or []
        if samples:
            print("Sample Evidence:")
            for sample in samples:
                print(
                    f"  - chunk_id={sample.get('chunk_id')}, "
                    f"score={sample.get('score')}, "
                    f"section={sample.get('section_title')}"
                )
                preview = sample.get("text_preview")
                if preview:
                    print(f"    preview={preview[:300]}")

    print("\nRecommended Actions:")
    recommended_actions = analysis.get("recommended_actions") or []

    if not recommended_actions:
        print("No recommended actions found.")
        return

    for action in recommended_actions:
        print(
            f"- {action.get('field_name')}: "
            f"{action.get('recommended_action')} "
            f"({action.get('failure_type')})"
        )


def print_llm_fallback(final_state: dict) -> None:
    print("\n" + "=" * 80)
    print("LLM Fallback")
    print("=" * 80)

    print(f"Attempted: {final_state.get('llm_fallback_attempted')}")
    print(f"Attempts: {final_state.get('llm_fallback_attempts')}")
    print(f"Fields: {final_state.get('llm_fallback_fields')}")

    llm_result = final_state.get("llm_fallback_result")

    if not llm_result:
        print("No LLM fallback result.")
        return

    print("\nLLM field results:")
    field_results = llm_result.get("llm_field_results", {})

    for field_name, result in field_results.items():
        print("\n" + "-" * 60)
        print(f"Field: {field_name}")
        print(f"Value: {result.get('value')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Evidence quote: {result.get('evidence_quote')}")
        print(f"Reason: {result.get('reason')}")


def print_plan(final_state: dict) -> None:
    print("\n" + "=" * 80)
    print("Event Plan")
    print("=" * 80)

    plan = final_state.get("plan", {})
    print(json.dumps(plan, indent=2))


def print_extracted_facts(final_state: dict) -> None:
    print("\n" + "=" * 80)
    print("Extracted Event Facts")
    print("=" * 80)

    confidence_result = final_state.get("confidence_result") or {}
    scored_fields = confidence_result.get("scored_fields") or {}

    if not scored_fields:
        print("No scored fields found.")
        return

    for field_name, field in scored_fields.items():
        print(f"\n{field_name}:")
        print(f"  value: {field.get('value')}")
        print(f"  extraction_method: {field.get('extraction_method')}")
        print(f"  extractor_confidence: {field.get('extractor_confidence')}")
        print(f"  evidence_quality_score: {field.get('evidence_quality_score')}")
        print(f"  final_confidence: {field.get('final_confidence')}")
        print(f"  quality_guard_status: {field.get('quality_guard_status')}")
        print(f"  reason: {field.get('confidence_reason') or field.get('reason')}")
        print(f"  evidence_chunk_ids: {field.get('evidence_chunk_ids')}")


def print_evidence_bundle(evidence_bundle: dict) -> None:
    for field_name, bundle in evidence_bundle.items():
        print("\n" + "=" * 80)
        print(f"FIELD: {field_name}")
        print("=" * 80)

        print(f"Query: {bundle['query']}")

        chunks = bundle["evidence_chunks"]

        if not chunks:
            print("No evidence chunks found.")
            continue

        for idx, chunk in enumerate(chunks, start=1):
            print("\n" + "-" * 60)
            print(f"Evidence #{idx}")
            print(f"Chunk ID: {chunk.get('id')}")
            print(f"Score: {chunk.get('score')}")
            print(f"Company: {chunk.get('company_name')}")
            print(f"Symbol: {chunk.get('symbol')}")
            print(f"Section: {chunk.get('section_title')}")

            text = chunk.get("chunk_text") or ""
            print("\nText:")
            print(text[:1200])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test EventLens AI LangGraph retrieval and analysis workflow"
    )

    parser.add_argument("--case-id", required=True, help="EventLens case ID")
    parser.add_argument("--event-type", required=True, help="Event type")
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print full evidence bundle chunks",
    )

    args = parser.parse_args()

    graph = build_eventlens_analysis_graph()

    initial_state = build_initial_state(
        case_id=args.case_id,
        event_type=args.event_type,
    )

    final_state = invoke_graph_with_trace(
        graph=graph,
        initial_state=initial_state,
        case_id=args.case_id,
        event_type=args.event_type,
    )

    print("\n" + "=" * 80)
    print("EventLens AI Retrieval / Analysis Test")
    print("=" * 80)

    print_agent_trace(final_state)
    print_autonomy_decision_history(final_state)
    print_summary(final_state)
    print_judge_result(final_state)
    print_recovery_metrics(final_state)
    print_evidence_gap_analysis(final_state)
    print_amount_recovery(final_state)
    print_llm_fallback(final_state)
    print_plan(final_state)
    print_extracted_facts(final_state)

    if args.show_evidence:
        print("\n" + "=" * 80)
        print("Evidence Bundle")
        print("=" * 80)

        evidence_bundle = final_state.get("evidence_bundle") or {}
        print_evidence_bundle(evidence_bundle)


if __name__ == "__main__":
    main()