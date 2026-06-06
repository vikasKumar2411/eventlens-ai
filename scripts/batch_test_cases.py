import csv
import json
import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.graph.workflow import build_eventlens_analysis_graph
from app.services.batch_failure_analyzer_service import BatchFailureAnalyzerService


TEST_CASES = [
    "EVL-2024-00003",
    "EVL-2024-00008",
    "EVL-2024-00011",
    # "EVL-2024-00012",
    # "EVL-2024-00004",
    # "EVL-2024-00014",
    # "EVL-2024-00006",
    # "EVL-2024-00015",
    # "EVL-2024-00002",
    # "EVL-2024-00001",
]

EVENT_TYPE = "debt_or_financing"

REQUIRED_FIELDS = [
    "borrower_or_issuer",
    "debt_type",
    "principal_amount",
    "maturity_date",
    "interest_rate",
    "use_of_proceeds",
    "lender_or_underwriter",
    "collateral_or_guarantee",
]


OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_OUTPUT_PATH = OUTPUT_DIR / "batch_eval_results.csv"
JSON_OUTPUT_PATH = OUTPUT_DIR / "batch_eval_results.json"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "batch_eval_summary.json"


def _get_score_by_stage(result: Dict[str, Any], stage_name: str) -> Optional[float]:
    history = result.get("judge_score_history") or []

    for item in history:
        if item.get("stage") == stage_name:
            score = item.get("judge_score")

            if isinstance(score, (int, float)):
                return float(score)

            if isinstance(score, str):
                try:
                    return float(score)
                except ValueError:
                    return None

    return None


def _get_initial_judge_score(result: Dict[str, Any]) -> Optional[float]:
    return _get_score_by_stage(result, "initial_judge")


def _get_post_llm_fallback_score(result: Dict[str, Any]) -> Optional[float]:
    return _get_score_by_stage(result, "post_llm_fallback")


def _get_post_retrieval_recovery_score(result: Dict[str, Any]) -> Optional[float]:
    return _get_score_by_stage(result, "post_retrieval_recovery")


def _score_delta(start: Optional[float], end: Optional[float]) -> Optional[float]:
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return round(end - start, 4)

    return None


def _get_overall_score_improvement(result: Dict[str, Any]) -> Optional[float]:
    return _score_delta(
        _get_initial_judge_score(result),
        _get_judge_score(result),
    )


def _get_llm_fallback_score_improvement(result: Dict[str, Any]) -> Optional[float]:
    return _score_delta(
        _get_initial_judge_score(result),
        _get_post_llm_fallback_score(result),
    )


def _get_retrieval_recovery_score_improvement(result: Dict[str, Any]) -> Optional[float]:
    return _score_delta(
        _get_pre_recovery_score(result),
        _get_post_recovery_score(result),
    )


def _get_score_from_judge_result(judge_result: Dict[str, Any]) -> Optional[float]:
    if not judge_result:
        return None

    score = judge_result.get("judge_score")

    if isinstance(score, (int, float)):
        return float(score)

    if isinstance(score, str):
        try:
            return float(score)
        except ValueError:
            return None

    return None


def _get_pre_recovery_score(result: Dict[str, Any]) -> Optional[float]:
    return _get_score_from_judge_result(
        result.get("pre_recovery_judge_result") or {}
    )


def _get_post_recovery_score(result: Dict[str, Any]) -> Optional[float]:
    return _get_score_from_judge_result(
        result.get("post_recovery_judge_result") or {}
    )


def _get_score_improvement(result: Dict[str, Any]) -> Optional[float]:
    pre = _get_pre_recovery_score(result)
    post = _get_post_recovery_score(result)

    if isinstance(pre, (int, float)) and isinstance(post, (int, float)):
        return round(post - pre, 4)

    return None


def _get_judge_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return result.get("judge_result") or {}


def _get_confidence_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return result.get("confidence_result") or {}


def _get_scored_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    confidence_result = _get_confidence_result(result)
    return confidence_result.get("scored_fields") or {}


def _get_field_value(result: Dict[str, Any], field_name: str) -> Optional[Any]:
    scored_fields = _get_scored_fields(result)
    field = scored_fields.get(field_name) or {}
    return field.get("value")


def _get_field_confidence(result: Dict[str, Any], field_name: str) -> Optional[Any]:
    scored_fields = _get_scored_fields(result)
    field = scored_fields.get(field_name) or {}
    return field.get("final_confidence")


def _get_judge_status(result: Dict[str, Any]) -> str:
    judge_result = _get_judge_result(result)
    return (
        judge_result.get("overall_status")
        or judge_result.get("status")
        or "unknown"
    )


def _get_judge_score(result: Dict[str, Any]) -> Optional[float]:
    return _get_score_from_judge_result(_get_judge_result(result))

def _get_should_recover(result: Dict[str, Any]) -> bool:
    judge_result = _get_judge_result(result)
    return bool(judge_result.get("should_recover") or False)


def _get_summary_confidence(result: Dict[str, Any]) -> Optional[Any]:
    summary_result = result.get("summary_result") or {}
    return summary_result.get("summary_confidence")


def _get_failed_checks(result: Dict[str, Any]) -> List[str]:
    judge_result = _get_judge_result(result)
    failed_checks = judge_result.get("failed_checks") or []

    return [
        check.get("name", "unknown_check")
        for check in failed_checks
        if isinstance(check, dict)
    ]


def _get_missing_fields_final(result: Dict[str, Any]) -> List[str]:
    missing = []

    for field_name in REQUIRED_FIELDS:
        value = _get_field_value(result, field_name)

        if value in [None, "", "unknown", "not_found", "N/A"]:
            missing.append(field_name)

    return missing


def _get_autonomy_decision_history(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    final_report = result.get("final_report") or {}

    return (
        final_report.get("autonomy_decision_history")
        or result.get("autonomy_decision_history")
        or []
    )


def _get_autonomy_actions(result: Dict[str, Any]) -> List[str]:
    history = _get_autonomy_decision_history(result)

    actions = []

    for decision in history:
        action = decision.get("selected_action")
        if action:
            actions.append(action)

    return actions


def _get_fields_recovered_by_llm(result: Dict[str, Any]) -> List[str]:
    llm_result = result.get("llm_fallback_result") or {}
    field_results = llm_result.get("llm_field_results") or {}

    recovered = []

    for field_name, field_result in field_results.items():
        value = field_result.get("value")
        confidence = field_result.get("confidence", 0)

        if value not in [None, "", "unknown", "not_found", "N/A"] and confidence >= 0.6:
            recovered.append(field_name)

    return recovered


def _get_preserved_fields(result: Dict[str, Any]) -> List[str]:
    preserved = result.get("preserved_recovered_fields") or {}

    if isinstance(preserved, dict):
        return list(preserved.keys())

    return []


def build_initial_state(case_id: str) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "event_type": EVENT_TYPE,
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

        # Future measurable recovery state
        "pre_recovery_judge_result": {},
        "post_llm_fallback_judge_result": {},
        "post_recovery_judge_result": {},
        "judge_score_history": [],
        "field_improvement_history": [],

        # Error state
        "errors": [],
    }


def run_case(graph, case_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 100)
    print(f"Running case: {case_id}")
    print("=" * 100)

    initial_state = build_initial_state(case_id)

    started_at = time.time()

    try:
        result = graph.invoke(initial_state)
        runtime_seconds = round(time.time() - started_at, 3)

        row = {
            "case_id": case_id,
            "event_type": EVENT_TYPE,
            "final_judge_score": _get_judge_score(result),
            "pre_recovery_judge_score": _get_pre_recovery_score(result),
            "post_recovery_judge_score": _get_post_recovery_score(result),
            "score_improvement": _get_score_improvement(result),
            "judge_score_history": result.get("judge_score_history") or [],
            "field_improvement_history": result.get("field_improvement_history") or [],
            "judge_status": _get_judge_status(result),
            "final_status": (result.get("final_report") or {}).get("status"),
            "summary_confidence": _get_summary_confidence(result),
            "should_recover": _get_should_recover(result),

            "initial_judge_score": _get_initial_judge_score(result),
            "post_llm_fallback_judge_score": _get_post_llm_fallback_score(result),
            "post_retrieval_recovery_judge_score": _get_post_retrieval_recovery_score(result),
            "overall_score_improvement": _get_overall_score_improvement(result),
            "llm_fallback_score_improvement": _get_llm_fallback_score_improvement(result),
            "retrieval_recovery_score_improvement": _get_retrieval_recovery_score_improvement(result),

            "recovery_attempts": result.get("recovery_attempts", 0),
            "llm_fallback_attempts": result.get("llm_fallback_attempts", 0),
            "autonomy_decision_count": len(_get_autonomy_decision_history(result)),
            "autonomy_actions": _get_autonomy_actions(result),

            "fields_recovered_by_llm": _get_fields_recovered_by_llm(result),
            "preserved_fields": _get_preserved_fields(result),
            "missing_fields_final": _get_missing_fields_final(result),

            "borrower_or_issuer": _get_field_value(result, "borrower_or_issuer"),
            "debt_type": _get_field_value(result, "debt_type"),
            "principal_amount": _get_field_value(result, "principal_amount"),
            "principal_confidence": _get_field_confidence(result, "principal_amount"),
            "maturity_date": _get_field_value(result, "maturity_date"),
            "maturity_confidence": _get_field_confidence(result, "maturity_date"),

            "amount_recovery_attempted": _get_amount_recovery_attempted(result),
            "amount_recovery_recovered": _get_amount_recovery_recovered(result),
            "amount_recovery_reason": _get_amount_recovery_reason(result),
            "amount_recovery_top_candidates": _get_amount_recovery_top_candidates(result),

            "failed_checks": _get_failed_checks(result),
            "runtime_seconds": runtime_seconds,
            "error": None,
        }

        print("\nCase result:")
        pprint(row)

        return row

    except Exception as exc:
        runtime_seconds = round(time.time() - started_at, 3)

        error_row = {
            "case_id": case_id,
            "event_type": EVENT_TYPE,

            "final_judge_score": None,
            "pre_recovery_judge_score": None,
            "post_recovery_judge_score": None,
            "score_improvement": None,
            "initial_judge_score": None,
            "post_llm_fallback_judge_score": None,
            "post_retrieval_recovery_judge_score": None,
            "overall_score_improvement": None,
            "llm_fallback_score_improvement": None,
            "retrieval_recovery_score_improvement": None,
            "judge_score_history": [],
            "field_improvement_history": [],
            "judge_status": "error",

            "final_status": "error",
            "summary_confidence": None,
            "should_recover": False,

            "recovery_attempts": None,
            "llm_fallback_attempts": None,
            "autonomy_decision_count": 0,
            "autonomy_actions": [],

            "fields_recovered_by_llm": [],
            "preserved_fields": [],
            "missing_fields_final": REQUIRED_FIELDS,

            "borrower_or_issuer": None,
            "debt_type": None,
            "principal_amount": None,
            "principal_confidence": None,
            "maturity_date": None,
            "maturity_confidence": None,

            "failed_checks": [],
            "runtime_seconds": runtime_seconds,
            "error": str(exc),
        }

        print("\nCase failed:")
        pprint(error_row)

        return error_row


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 180)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 180)

    print(
        f"{'case_id':<18} "
        f"{'status':<10} "
        f"{'initial':<10} "
        f"{'post_llm':<10} "
        f"{'post_ret':<10} "
        f"{'final':<10} "
        f"{'overall_Δ':<10} "
        f"{'llm_Δ':<10} "
        f"{'ret_Δ':<10} "
        f"{'rec_attempts':<14} "
        f"{'auto_decisions':<16} "
        f"{'runtime':<10}"
    )

    print("-" * 180)

    for row in rows:
        print(
            f"{str(row.get('case_id')):<18} "
            f"{str(row.get('judge_status')):<10} "
            f"{str(row.get('initial_judge_score')):<10} "
            f"{str(row.get('post_llm_fallback_judge_score')):<10} "
            f"{str(row.get('post_retrieval_recovery_judge_score')):<10} "
            f"{str(row.get('final_judge_score')):<10} "
            f"{str(row.get('overall_score_improvement')):<10} "
            f"{str(row.get('llm_fallback_score_improvement')):<10} "
            f"{str(row.get('retrieval_recovery_score_improvement')):<10} "
            f"{str(row.get('recovery_attempts')):<14} "
            f"{str(row.get('autonomy_decision_count')):<16} "
            f"{str(row.get('runtime_seconds')):<10}"
        )


def build_batch_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("judge_status") == "pass")
    failed = sum(1 for row in rows if row.get("judge_status") == "fail")
    errored = sum(1 for row in rows if row.get("judge_status") == "error")

    scores = [
        row.get("final_judge_score")
        for row in rows
        if isinstance(row.get("final_judge_score"), (int, float))
    ]

    avg_score = round(sum(scores) / len(scores), 4) if scores else None

    overall_improvements = [
        row.get("overall_score_improvement")
        for row in rows
        if isinstance(row.get("overall_score_improvement"), (int, float))
    ]

    llm_improvements = [
        row.get("llm_fallback_score_improvement")
        for row in rows
        if isinstance(row.get("llm_fallback_score_improvement"), (int, float))
    ]

    retrieval_improvements = [
        row.get("retrieval_recovery_score_improvement")
        for row in rows
        if isinstance(row.get("retrieval_recovery_score_improvement"), (int, float))
    ]

    avg_overall_improvement = (
        round(sum(overall_improvements) / len(overall_improvements), 4)
        if overall_improvements
        else None
    )

    avg_llm_improvement = (
        round(sum(llm_improvements) / len(llm_improvements), 4)
        if llm_improvements
        else None
    )

    avg_retrieval_improvement = (
        round(sum(retrieval_improvements) / len(retrieval_improvements), 4)
        if retrieval_improvements
        else None
    )

    total_recovered_fields = sum(
        len(row.get("fields_recovered_by_llm") or [])
        for row in rows
    )

    total_missing_fields = sum(
        len(row.get("missing_fields_final") or [])
        for row in rows
    )

    total_autonomy_decisions = sum(
        row.get("autonomy_decision_count") or 0
        for row in rows
    )

    llm_attempt_cases = sum(
        1 for row in rows
        if (row.get("llm_fallback_attempts") or 0) > 0
    )

    return {
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "pass_rate": round((passed / total) * 100, 2) if total else 0,
        "average_final_judge_score": avg_score,
        "average_overall_score_improvement": avg_overall_improvement,
        "average_llm_fallback_score_improvement": avg_llm_improvement,
        "average_retrieval_recovery_score_improvement": avg_retrieval_improvement,
        "total_fields_recovered_by_llm": total_recovered_fields,
        "total_missing_fields_final": total_missing_fields,
        "total_autonomy_decisions": total_autonomy_decisions,
        "cases_with_llm_fallback": llm_attempt_cases,
    }



def print_failure_analysis(rows: List[Dict[str, Any]]) -> None:
    summary = build_batch_summary(rows)

    print("\n" + "=" * 120)
    print("FAILURE ANALYSIS")
    print("=" * 120)

    pprint(summary)

    failed_rows = [
        row for row in rows
        if row.get("judge_status") in {"fail", "error"}
    ]

    if not failed_rows:
        print("\nNo failures found.")
        return

    print("\nFailed/error cases:")
    for row in failed_rows:
        print(
            f"- {row.get('case_id')}: "
            f"status={row.get('judge_status')}, "
            f"score={row.get('final_judge_score')}, "
            f"missing={row.get('missing_fields_final')}, "
            f"failed_checks={row.get('failed_checks')}, "
            f"error={row.get('error')}"
        )


def print_autonomous_batch_analysis(rows: List[Dict[str, Any]]) -> None:
    analyzer = BatchFailureAnalyzerService()
    batch_analysis = analyzer.analyze(rows)

    print("\n" + "=" * 120)
    print("AUTONOMOUS BATCH FAILURE ANALYSIS")
    print("=" * 120)

    pprint(batch_analysis)


def _get_amount_recovery_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return result.get("amount_recovery_result") or {}


def _get_amount_recovery_attempted(result: Dict[str, Any]) -> bool:
    return bool(_get_amount_recovery_result(result))


def _get_amount_recovery_recovered(result: Dict[str, Any]) -> bool:
    return bool(_get_amount_recovery_result(result).get("recovered"))


def _get_amount_recovery_reason(result: Dict[str, Any]) -> Optional[str]:
    return _get_amount_recovery_result(result).get("reason")


def _get_amount_recovery_top_candidates(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (_get_amount_recovery_result(result).get("candidates") or [])[:3]


def save_outputs(rows: List[Dict[str, Any]]) -> None:
    summary = build_batch_summary(rows)

    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    with open(SUMMARY_OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if rows:
        fieldnames = list(rows[0].keys())

        with open(CSV_OUTPUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                csv_row = {}

                for key, value in row.items():
                    if isinstance(value, (list, dict)):
                        csv_row[key] = json.dumps(value, default=str)
                    else:
                        csv_row[key] = value

                writer.writerow(csv_row)

    print("\nSaved outputs:")
    print(f"- {CSV_OUTPUT_PATH}")
    print(f"- {JSON_OUTPUT_PATH}")
    print(f"- {SUMMARY_OUTPUT_PATH}")


def main():
    graph = build_eventlens_analysis_graph()

    rows = []

    for case_id in TEST_CASES:
        row = run_case(graph, case_id)
        rows.append(row)

    print_summary_table(rows)
    print_failure_analysis(rows)
    print_autonomous_batch_analysis(rows)
    save_outputs(rows)


if __name__ == "__main__":
    main()