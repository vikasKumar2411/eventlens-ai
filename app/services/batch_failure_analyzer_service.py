from collections import Counter, defaultdict
from typing import Any, Dict, List


class BatchFailureAnalyzerService:
    """
    Analyzes batch evaluation rows and identifies repeated failure patterns.

    This is the first autonomous meta-evaluation layer:
    the system evaluates not only cases, but also its own failure modes.
    """

    def analyze(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_cases = len(rows)
        passed_rows = [row for row in rows if row.get("judge_status") == "pass"]
        failed_rows = [row for row in rows if row.get("judge_status") == "fail"]
        error_rows = [row for row in rows if row.get("judge_status") == "error"]

        failed_check_counter = Counter()
        weak_field_counter = Counter()
        missing_field_counter = Counter()
        suspicious_value_counter = Counter()

        for row in failed_rows:
            for check_name in row.get("failed_checks", []):
                failed_check_counter[check_name] += 1

            self._inspect_row_for_field_issues(
                row=row,
                weak_field_counter=weak_field_counter,
                missing_field_counter=missing_field_counter,
                suspicious_value_counter=suspicious_value_counter,
            )

        pass_rate = round((len(passed_rows) / total_cases) * 100, 2) if total_cases else 0.0

        recommendations = self._build_recommendations(
            failed_check_counter=failed_check_counter,
            weak_field_counter=weak_field_counter,
            missing_field_counter=missing_field_counter,
            suspicious_value_counter=suspicious_value_counter,
        )

        return {
            "total_cases": total_cases,
            "passed": len(passed_rows),
            "failed": len(failed_rows),
            "errored": len(error_rows),
            "pass_rate": pass_rate,
            "common_failed_checks": dict(failed_check_counter),
            "weak_fields": dict(weak_field_counter),
            "missing_fields": dict(missing_field_counter),
            "suspicious_values": dict(suspicious_value_counter),
            "recommendations": recommendations,
        }

    def _inspect_row_for_field_issues(
        self,
        row: Dict[str, Any],
        weak_field_counter: Counter,
        missing_field_counter: Counter,
        suspicious_value_counter: Counter,
    ) -> None:
        field_names = [
            "principal_amount",
            "maturity_date",
            "borrower_or_issuer",
            "debt_type",
        ]

        for field_name in field_names:
            value = row.get(field_name)

            if value is None:
                missing_field_counter[field_name] += 1

        principal_amount = row.get("principal_amount")
        principal_confidence = row.get("principal_confidence")

        if principal_confidence is not None and principal_confidence < 0.7:
            weak_field_counter["principal_amount"] += 1

        if principal_amount and self._looks_like_suspicious_principal(principal_amount):
            suspicious_value_counter["principal_amount"] += 1

        maturity_date = row.get("maturity_date")
        maturity_confidence = row.get("maturity_confidence")

        if maturity_confidence is not None and maturity_confidence < 0.7:
            weak_field_counter["maturity_date"] += 1

        if maturity_date is None:
            missing_field_counter["maturity_date"] += 1

    def _looks_like_suspicious_principal(self, value: Any) -> bool:
        text = str(value)

        # Very rough MVP rule:
        # principal amounts for debt financing are usually larger and often have
        # full dollar formatting. Small values like $2,000 or $1,297.9 are suspicious.
        normalized = (
            text.replace("$", "")
            .replace(",", "")
            .strip()
        )

        try:
            amount = float(normalized)
        except ValueError:
            return False

        return amount < 1_000_000

    def _build_recommendations(
        self,
        failed_check_counter: Counter,
        weak_field_counter: Counter,
        missing_field_counter: Counter,
        suspicious_value_counter: Counter,
    ) -> List[str]:
        recommendations = []

        if failed_check_counter.get("critical_field_confidence", 0) > 0:
            recommendations.append(
                "Improve field-level extraction and confidence scoring for critical fields."
            )

        if failed_check_counter.get("summary_grounding", 0) > 0:
            recommendations.append(
                "Make summary generation stricter: only include fields with evidence and sufficient confidence."
            )

        if missing_field_counter.get("maturity_date", 0) > 0:
            recommendations.append(
                "Add a stronger maturity_date extractor that prioritizes phrases like 'matures on', 'maturity date', and 'due'."
            )

        if weak_field_counter.get("principal_amount", 0) > 0:
            recommendations.append(
                "Improve principal_amount extraction by requiring nearby debt terms such as 'principal amount', 'commitment', 'facility', 'notes', or 'loans'."
            )

        if suspicious_value_counter.get("principal_amount", 0) > 0:
            recommendations.append(
                "Reject suspicious principal_amount candidates below $1,000,000 unless supported by strong financing context."
            )

        if not recommendations:
            recommendations.append("No repeated failure pattern detected.")

        return recommendations