from typing import Any, Dict, List


REQUIRED_DEBT_OR_FINANCING_FIELDS = [
    "borrower_or_issuer",
    "debt_type",
    "principal_amount",
    "maturity_date",
    "interest_rate",
    "use_of_proceeds",
    "lender_or_underwriter",
    "collateral_or_guarantee",
]


EVIDENCE_SENSITIVE_FIELDS = {
    "principal_amount",
    "debt_type",
    "maturity_date",
    "interest_rate",
}


class RecoveryPlannerService:
    """
    Creates a bounded recovery plan based on judge failures and missing/weak fields.

    The planner decomposes the recovery goal into ordered recovery actions.
    It does not execute anything directly.
    """

    def create_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        judge_result = state.get("judge_result") or {}
        extraction_result = state.get("extraction_result") or {}
        confidence_result = state.get("confidence_result") or {}

        missing_fields = self._get_missing_fields(
            extraction_result=extraction_result,
            confidence_result=confidence_result,
        )

        weak_fields = self._get_weak_fields(judge_result)

        target_fields = list(dict.fromkeys(missing_fields + weak_fields))

        steps: List[Dict[str, Any]] = []

        if target_fields:
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "action": "llm_extraction_fallback",
                    "target_fields": target_fields,
                    "reason": "Recover missing or weak fields using fallback extraction.",
                }
            )

        evidence_sensitive_fields = [
            field for field in target_fields
            if field in EVIDENCE_SENSITIVE_FIELDS
        ]

        if evidence_sensitive_fields:
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "action": "evidence_gap_analysis",
                    "target_fields": evidence_sensitive_fields,
                    "reason": "Diagnose whether weak fields lack supporting evidence.",
                }
            )

        if "principal_amount" in target_fields:
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "action": "amount_recovery",
                    "target_fields": ["principal_amount"],
                    "reason": "Use specialist amount recovery for principal_amount.",
                }
            )

        if evidence_sensitive_fields:
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "action": "retry_retrieval",
                    "target_fields": evidence_sensitive_fields,
                    "reason": "Retry retrieval for fields with insufficient or weak evidence.",
                }
            )

        steps.append(
            {
                "step_id": len(steps) + 1,
                "action": "rerun_judge",
                "target_fields": target_fields,
                "reason": "Re-evaluate after recovery actions.",
            }
        )

        return {
            "goal": "Recover weak or missing debt financing fields with grounded evidence.",
            "target_fields": target_fields,
            "steps": steps,
            "stop_conditions": [
                "judge_passes",
                "max_steps_reached",
                "no_score_improvement",
                "remaining_fields_lack_evidence",
                "no_valid_steps_remaining",
            ],
            "planner_type": "deterministic_bounded_recovery_planner",
        }

    def _get_missing_fields(
        self,
        extraction_result: Dict[str, Any],
        confidence_result: Dict[str, Any],
    ) -> List[str]:
        """
        Detects missing fields from confidence_result first, then falls back to
        extraction_result.

        In EventLens, fields may appear directly under extraction_result or under
        confidence_result["scored_fields"].
        """

        missing_fields = []

        scored_fields = confidence_result.get("scored_fields") or {}

        for field_name in REQUIRED_DEBT_OR_FINANCING_FIELDS:
            field = scored_fields.get(field_name)

            if isinstance(field, dict):
                value = field.get("value")
            else:
                raw_field = extraction_result.get(field_name)

                if isinstance(raw_field, dict):
                    value = raw_field.get("value")
                else:
                    value = raw_field

            if value in (None, "", "unknown", "not_found", "N/A"):
                missing_fields.append(field_name)

        return missing_fields

    def _get_weak_fields(self, judge_result: Dict[str, Any]) -> List[str]:
        weak_fields = []

        field_scores = judge_result.get("field_scores") or {}

        for field_name, score_info in field_scores.items():
            if isinstance(score_info, dict):
                score = (
                    score_info.get("score")
                    or score_info.get("judge_score")
                    or score_info.get("confidence")
                )
            else:
                score = None

            if score is not None and score < 0.75:
                weak_fields.append(field_name)

        failed_fields = judge_result.get("failed_fields") or []
        weak_fields.extend(failed_fields)

        weak_or_missing_fields = judge_result.get("weak_or_missing_fields") or []
        weak_fields.extend(weak_or_missing_fields)

        fields_to_recover = judge_result.get("fields_to_recover") or []
        weak_fields.extend(fields_to_recover)

        return [
            field for field in dict.fromkeys(weak_fields)
            if field in REQUIRED_DEBT_OR_FINANCING_FIELDS
        ]