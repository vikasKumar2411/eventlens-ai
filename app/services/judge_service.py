from typing import Any, Dict, List


class JudgeService:
    def evaluate(
        self,
        plan: Dict[str, Any],
        confidence_result: Dict[str, Any],
        summary_result: Dict[str, Any],
        evidence_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        required_fields = plan.get("required_fields", [])
        scored_fields = confidence_result.get("scored_fields", {})
        summary = summary_result.get("summary")

        checks = []

        checks.append(
            self._check_required_fields_present(
                required_fields=required_fields,
                scored_fields=scored_fields,
            )
        )

        checks.append(
            self._check_critical_field_confidence(
                event_type=plan.get("event_type"),
                scored_fields=scored_fields,
            )
        )

        checks.append(
            self._check_summary_exists(summary=summary)
        )

        checks.append(
            self._check_summary_grounding(
                summary=summary,
                scored_fields=scored_fields,
            )
        )

        checks.append(
            self._check_evidence_availability(
                required_fields=required_fields,
                evidence_bundle=evidence_bundle,
            )
        )

        failed_checks = [check for check in checks if check["status"] == "fail"]
        warning_checks = [check for check in checks if check["status"] == "warning"]

        if failed_checks:
            overall_status = "fail"
            should_recover = True
        elif warning_checks:
            overall_status = "warning"
            should_recover = False
        else:
            overall_status = "pass"
            should_recover = False

        judge_score = self._calculate_judge_score(checks)

        return {
            "case_id": plan.get("case_id"),
            "event_type": plan.get("event_type"),
            "overall_status": overall_status,
            "judge_score": judge_score,
            "should_recover": should_recover,
            "checks": checks,
            "failed_checks": failed_checks,
            "warning_checks": warning_checks,
        }

    def _check_required_fields_present(
        self,
        required_fields: List[str],
        scored_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        missing = []

        for field_name in required_fields:
            field = scored_fields.get(field_name, {})
            if not field.get("value"):
                missing.append(field_name)

        if missing:
            return {
                "name": "required_fields_present",
                "status": "warning",
                "score": 0.70,
                "message": f"Missing extracted values for fields: {missing}",
                "details": {"missing_fields": missing},
            }

        return {
            "name": "required_fields_present",
            "status": "pass",
            "score": 1.0,
            "message": "All required fields have extracted values.",
            "details": {},
        }

    def _check_critical_field_confidence(
        self,
        event_type: str,
        scored_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        critical_fields_by_event_type = {
            "debt_or_financing": [
                "borrower_or_issuer",
                "debt_type",
                "principal_amount",
                "maturity_date",
                "lender_or_underwriter",
            ]
        }

        critical_fields = critical_fields_by_event_type.get(event_type, [])

        low_confidence = []

        for field_name in critical_fields:
            field = scored_fields.get(field_name, {})
            final_confidence = float(field.get("final_confidence") or 0.0)

            if final_confidence < 0.70:
                low_confidence.append(
                    {
                        "field_name": field_name,
                        "value": field.get("value"),
                        "final_confidence": final_confidence,
                    }
                )

        if low_confidence:
            return {
                "name": "critical_field_confidence",
                "status": "fail",
                "score": 0.40,
                "message": "One or more critical fields have low confidence.",
                "details": {"low_confidence_fields": low_confidence},
            }

        return {
            "name": "critical_field_confidence",
            "status": "pass",
            "score": 1.0,
            "message": "Critical fields meet confidence threshold.",
            "details": {},
        }

    def _check_summary_exists(
        self,
        summary: str,
    ) -> Dict[str, Any]:
        if not summary:
            return {
                "name": "summary_exists",
                "status": "fail",
                "score": 0.0,
                "message": "No summary was generated.",
                "details": {},
            }

        if len(summary.split()) < 20:
            return {
                "name": "summary_exists",
                "status": "warning",
                "score": 0.70,
                "message": "Summary exists but is very short.",
                "details": {"word_count": len(summary.split())},
            }

        return {
            "name": "summary_exists",
            "status": "pass",
            "score": 1.0,
            "message": "Summary was generated.",
            "details": {"word_count": len(summary.split())},
        }

    def _check_summary_grounding(
        self,
        summary: str,
        scored_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not summary:
            return {
                "name": "summary_grounding",
                "status": "fail",
                "score": 0.0,
                "message": "Cannot evaluate grounding because summary is missing.",
                "details": {},
            }

        summary_lower = summary.lower()

        important_fields = [
            "borrower_or_issuer",
            "debt_type",
            "principal_amount",
            "maturity_date",
            "lender_or_underwriter",
            "use_of_proceeds",
        ]

        supported = []
        unsupported = []

        for field_name in important_fields:
            field = scored_fields.get(field_name, {})
            value = field.get("value")

            if not value:
                unsupported.append(field_name)
                continue

            value_lower = str(value).lower()

            # Full string match can be too strict for long values, so also check meaningful tokens.
            meaningful_tokens = [
                token.strip(".,;:$()")
                for token in value_lower.replace("/", " ").replace(";", " ").split()
                if len(token.strip(".,;:$()")) >= 4
            ]

            has_full_match = value_lower in summary_lower
            has_token_match = any(token in summary_lower for token in meaningful_tokens[:5])

            if has_full_match or has_token_match:
                supported.append(field_name)
            else:
                unsupported.append(field_name)

        grounding_ratio = len(supported) / len(important_fields)

        if grounding_ratio >= 0.80:
            return {
                "name": "summary_grounding",
                "status": "pass",
                "score": round(grounding_ratio, 2),
                "message": "Summary is grounded in extracted fields.",
                "details": {
                    "supported_fields": supported,
                    "unsupported_fields": unsupported,
                },
            }

        if grounding_ratio >= 0.60:
            return {
                "name": "summary_grounding",
                "status": "warning",
                "score": round(grounding_ratio, 2),
                "message": "Summary is partially grounded but missing some important field references.",
                "details": {
                    "supported_fields": supported,
                    "unsupported_fields": unsupported,
                },
            }

        return {
            "name": "summary_grounding",
            "status": "fail",
            "score": round(grounding_ratio, 2),
            "message": "Summary is weakly grounded in extracted fields.",
            "details": {
                "supported_fields": supported,
                "unsupported_fields": unsupported,
            },
        }

    def _check_evidence_availability(
        self,
        required_fields: List[str],
        evidence_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        fields_without_evidence = []

        for field_name in required_fields:
            chunks = evidence_bundle.get(field_name, {}).get("evidence_chunks", [])
            if not chunks:
                fields_without_evidence.append(field_name)

        if fields_without_evidence:
            return {
                "name": "evidence_availability",
                "status": "fail",
                "score": 0.30,
                "message": "Some required fields have no retrieved evidence.",
                "details": {"fields_without_evidence": fields_without_evidence},
            }

        return {
            "name": "evidence_availability",
            "status": "pass",
            "score": 1.0,
            "message": "All required fields have retrieved evidence chunks.",
            "details": {},
        }

    def _calculate_judge_score(
        self,
        checks: List[Dict[str, Any]],
    ) -> float:
        if not checks:
            return 0.0

        scores = [float(check.get("score") or 0.0) for check in checks]
        return round(sum(scores) / len(scores), 2)