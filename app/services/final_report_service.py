from typing import Any, Dict


class FinalReportService:
    """
    Builds the final EventLens report.
    """

    def build_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        extraction_result = self._merge_preserved_recovered_fields(
            result=state.get("extraction_result", {}),
            preserved_recovered_fields=state.get("preserved_recovered_fields", {}),
        )

        confidence_result = self._merge_preserved_recovered_fields(
            result=state.get("confidence_result", {}),
            preserved_recovered_fields=state.get("preserved_recovered_fields", {}),
        )

        return {
            "case_id": state.get("case_id"),
            "event_type": state.get("event_type"),
            "status": self._get_status(state),
            "plan": state.get("plan"),
            "extraction_result": extraction_result,
            "confidence_result": confidence_result,
            "summary_result": state.get("summary_result"),
            "judge_result": state.get("judge_result"),
            "recovery_result": state.get("recovery_result"),
            "preserved_recovered_fields": state.get("preserved_recovered_fields", {}),
            "llm_fallback_result": state.get("llm_fallback_result"),
            "autonomy_decision_history": state.get("autonomy_decision_history", []),
            "agent_trace": state.get("agent_trace", []),
            "errors": state.get("errors", []),
        }

    def _get_status(self, state: Dict[str, Any]) -> str:
        judge_result = state.get("judge_result") or {}

        if judge_result.get("overall_status") == "pass" or judge_result.get("status") == "pass":
            return "pass"

        if state.get("recovery_result"):
            return "completed_after_recovery"

        if state.get("errors"):
            return "completed_with_errors"

        return "completed"

    def _merge_preserved_recovered_fields(
        self,
        result: Dict[str, Any],
        preserved_recovered_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ensures high-confidence LLM fallback fields survive into the final report.

        Rule:
        - Keep preserved field if current field is missing.
        - Keep preserved field if preserved confidence >= current confidence.
        """

        merged_result = dict(result or {})

        for field_name, preserved_field in (preserved_recovered_fields or {}).items():
            if not isinstance(preserved_field, dict):
                continue

            preserved_value = preserved_field.get("value")
            preserved_confidence = (
                preserved_field.get("final_confidence")
                or preserved_field.get("extractor_confidence")
                or 0.0
            )

            if preserved_value in (None, "", "N/A"):
                continue

            current_field = merged_result.get(field_name)

            if not isinstance(current_field, dict):
                merged_result[field_name] = preserved_field
                continue

            current_value = current_field.get("value")
            current_confidence = (
                current_field.get("final_confidence")
                or current_field.get("extractor_confidence")
                or 0.0
            )

            current_missing = current_value in (None, "", "N/A")
            preserved_is_better_or_equal = preserved_confidence >= current_confidence

            if current_missing or preserved_is_better_or_equal:
                merged_result[field_name] = preserved_field

        return merged_result