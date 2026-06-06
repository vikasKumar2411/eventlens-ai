from typing import Any, Dict, List
import json
import urllib.error
import urllib.request

from app.config.settings import settings


class LLMRecoverySupervisorService:
    """
    Decides the best recovery action after judge failure.

    This is the bounded autonomous recovery planner.
    It does not execute recovery itself.
    It only chooses the next recovery strategy.

    Uses local Ollama first.
    Falls back to deterministic policy if Ollama fails or returns invalid JSON.
    """

    ALLOWED_RECOVERY_ACTIONS = {
        "llm_extraction_fallback",
        "retry_retrieval",
        "final_report",
    }

    def _compact_field_snapshot(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds a compact field-level snapshot for the Ollama recovery supervisor prompt.

        Supports:
        - extraction_result
        - confidence_result
        - confidence_result["scored_fields"]
        - direct field maps
        """

        field_map = self._get_field_map(result)

        compact = {}

        for field_name, field_info in field_map.items():
            if not isinstance(field_info, dict):
                compact[field_name] = {
                    "value": field_info,
                }
                continue

            compact[field_name] = {
                "value": field_info.get("value"),
                "extraction_method": field_info.get("extraction_method"),
                "extractor_confidence": field_info.get("extractor_confidence"),
                "final_confidence": (
                    field_info.get("final_confidence")
                    or field_info.get("confidence")
                ),
                "quality_guard_status": field_info.get("quality_guard_status"),
            }

        return compact

    def decide_recovery_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        missing_fields = self._get_missing_fields(state.get("extraction_result", {}))
        low_confidence_fields = self._get_low_confidence_fields(
            state.get("confidence_result", {})
        )

        fallback_decision = self._deterministic_recovery_decision(
            state=state,
            missing_fields=missing_fields,
            low_confidence_fields=low_confidence_fields,
        )

        try:
            llm_decision = self._call_ollama_recovery_supervisor(
                state=state,
                missing_fields=missing_fields,
                low_confidence_fields=low_confidence_fields,
            )

            return self._validate_decision(
                decision=llm_decision,
                state=state,
                fallback_decision=fallback_decision,
                missing_fields=missing_fields,
                low_confidence_fields=low_confidence_fields,
            )

        except Exception as exc:
            fallback_decision["reason"] = (
                f"Ollama recovery supervisor failed, so deterministic fallback was used. "
                f"Original reason: {fallback_decision.get('reason')}. "
                f"Error: {str(exc)}"
            )
            return fallback_decision

    def _call_ollama_recovery_supervisor(
        self,
        state: Dict[str, Any],
        missing_fields: List[str],
        low_confidence_fields: List[str],
    ) -> Dict[str, Any]:
        prompt = self._build_recovery_prompt(
            state=state,
            missing_fields=missing_fields,
            low_confidence_fields=low_confidence_fields,
        )

        payload = {
            "model": settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the EventLens Recovery Supervisor. "
                        "You choose one bounded recovery action after judge failure. "
                        "Return only valid JSON. No markdown. No prose outside JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": settings.llm_temperature,
            },
        }

        url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=settings.llm_timeout_seconds,
        ) as response:
            raw_response = response.read().decode("utf-8")

        response_json = json.loads(raw_response)
        content = response_json.get("message", {}).get("content", "")

        if not content:
            raise ValueError("Ollama returned empty message content.")

        return json.loads(content)

    def _build_recovery_prompt(
        self,
        state: Dict[str, Any],
        missing_fields: List[str],
        low_confidence_fields: List[str],
    ) -> str:
        judge_result = state.get("judge_result", {})
        confidence_result = state.get("confidence_result", {})
        extraction_result = state.get("extraction_result", {})
        recovery_history = state.get("recovery_history", [])

        llm_fallback_attempts = state.get("llm_fallback_attempts", 0)
        max_llm_fallback_attempts = state.get("max_llm_fallback_attempts", 1)

        recovery_attempts = state.get("recovery_attempts", 0)
        max_recovery_attempts = state.get("max_recovery_attempts", 1)

        llm_recovery_decision_count = state.get("llm_recovery_decision_count", 0)
        max_llm_recovery_decisions = state.get("max_llm_recovery_decisions", 2)

        preserved_recovered_fields = state.get("preserved_recovered_fields", {})

        llm_fallback_available = llm_fallback_attempts < max_llm_fallback_attempts
        retry_retrieval_available = recovery_attempts < max_recovery_attempts

        compact_context = {
            "event_type": state.get("event_type"),
            "judge_result": {
                "overall_status": judge_result.get("overall_status")
                or judge_result.get("status"),
                "judge_score": judge_result.get("judge_score"),
                "should_recover": judge_result.get("should_recover"),
                "failed_checks": judge_result.get("failed_checks", []),
            },
            "missing_fields": missing_fields,
            "low_confidence_fields": low_confidence_fields,
            "available_actions": {
                "llm_extraction_fallback": llm_fallback_available,
                "retry_retrieval": retry_retrieval_available,
                "final_report": True,
            },
            "attempt_limits": {
                "llm_fallback_attempts": llm_fallback_attempts,
                "max_llm_fallback_attempts": max_llm_fallback_attempts,
                "recovery_attempts": recovery_attempts,
                "max_recovery_attempts": max_recovery_attempts,
                "llm_recovery_decision_count": llm_recovery_decision_count,
                "max_llm_recovery_decisions": max_llm_recovery_decisions,
            },
            "preserved_recovered_fields": {
                field_name: {
                    "value": field_info.get("value"),
                    "final_confidence": field_info.get("final_confidence"),
                    "extraction_method": field_info.get("extraction_method"),
                }
                for field_name, field_info in preserved_recovered_fields.items()
                if isinstance(field_info, dict)
            },
            "recovery_history": recovery_history,
            "extraction_snapshot": self._compact_field_snapshot(extraction_result),
            "confidence_snapshot": self._compact_field_snapshot(confidence_result),
        }

        return f"""
    You are the EventLens Recovery Supervisor.

    Your job:
    Choose exactly one next recovery action after judge failure.

    Allowed actions:
    1. "llm_extraction_fallback"
    2. "retry_retrieval"
    3. "final_report"

    Hard constraints:
    - You MUST NOT choose "llm_extraction_fallback" if llm_fallback_attempts >= max_llm_fallback_attempts.
    - You MUST NOT choose "retry_retrieval" if recovery_attempts >= max_recovery_attempts.
    - You MAY always choose "final_report".
    - Choose only an action where available_actions[action] is true.
    - Do not target fields that are already preserved with high confidence unless they are still failing.
    - Target only missing or low-confidence fields.

    Decision policy:
    - If missing or low-confidence fields remain and llm_extraction_fallback is available, choose "llm_extraction_fallback".
    - If missing or low-confidence fields remain and llm_extraction_fallback is exhausted and retry_retrieval is available, choose "retry_retrieval".
    - If all recovery actions are exhausted, choose "final_report".
    - If no useful recovery is possible, choose "final_report".

    Return strict JSON only.
    No markdown.
    No prose outside JSON.

    Required JSON schema:
    {{
    "next_action": "llm_extraction_fallback | retry_retrieval | final_report",
    "target_fields": ["field_name_1", "field_name_2"],
    "failure_mode": "weak_extraction | retrieval_insufficient | recovery_exhausted | no_recovery_needed",
    "reason": "short explanation"
    }}

    Context:
    {json.dumps(compact_context, indent=2)}
    """.strip()

    def _deterministic_recovery_decision(
        self,
        state: Dict[str, Any],
        missing_fields: List[str],
        low_confidence_fields: List[str],
    ) -> Dict[str, Any]:
        llm_fallback_attempts = state.get("llm_fallback_attempts", 0)
        max_llm_fallback_attempts = state.get("max_llm_fallback_attempts", 1)

        recovery_attempts = state.get("recovery_attempts", 0)
        max_recovery_attempts = state.get("max_recovery_attempts", 1)

        target_fields = self._dedupe_fields(missing_fields + low_confidence_fields)

        if target_fields:
            if llm_fallback_attempts < max_llm_fallback_attempts:
                return {
                    "next_action": "llm_extraction_fallback",
                    "target_fields": target_fields,
                    "failure_mode": "weak_extraction",
                    "reason": (
                        "Required or low-confidence fields remain after judge evaluation. "
                        "Evidence may exist, so LLM fallback should attempt targeted extraction."
                    ),
                }

            if recovery_attempts < max_recovery_attempts:
                return {
                    "next_action": "retry_retrieval",
                    "target_fields": target_fields,
                    "failure_mode": "retrieval_insufficient",
                    "reason": (
                        "LLM fallback is exhausted and fields are still missing or weak. "
                        "Retry retrieval for targeted fields."
                    ),
                }

        return {
            "next_action": "final_report",
            "target_fields": [],
            "failure_mode": "recovery_exhausted",
            "reason": (
                "Recovery options are exhausted or no clear recoverable fields remain. "
                "Finalize with available evidence and warnings."
            ),
        }

    def _validate_decision(
        self,
        decision: Dict[str, Any],
        state: Dict[str, Any],
        fallback_decision: Dict[str, Any],
        missing_fields: List[str],
        low_confidence_fields: List[str],
    ) -> Dict[str, Any]:
        if not isinstance(decision, dict):
            return fallback_decision

        next_action = decision.get("next_action")

        if next_action not in self.ALLOWED_RECOVERY_ACTIONS:
            fallback_decision["reason"] = (
                f"Ollama returned invalid recovery action: {next_action}. "
                f"Using deterministic fallback. {fallback_decision.get('reason')}"
            )
            return fallback_decision

        llm_fallback_attempts = state.get("llm_fallback_attempts", 0)
        max_llm_fallback_attempts = state.get("max_llm_fallback_attempts", 1)

        recovery_attempts = state.get("recovery_attempts", 0)
        max_recovery_attempts = state.get("max_recovery_attempts", 1)

        if (
            next_action == "llm_extraction_fallback"
            and llm_fallback_attempts >= max_llm_fallback_attempts
        ):
            fallback_decision["reason"] = (
                "Ollama selected llm_extraction_fallback, but fallback attempts are exhausted. "
                f"Using deterministic fallback. {fallback_decision.get('reason')}"
            )
            return fallback_decision

        if (
            next_action == "retry_retrieval"
            and recovery_attempts >= max_recovery_attempts
        ):
            fallback_decision["reason"] = (
                "Ollama selected retry_retrieval, but recovery attempts are exhausted. "
                f"Using deterministic fallback. {fallback_decision.get('reason')}"
            )
            return fallback_decision

        target_fields = decision.get("target_fields", [])

        if not isinstance(target_fields, list):
            target_fields = []

        allowed_target_fields = set(missing_fields + low_confidence_fields)

        if allowed_target_fields:
            target_fields = [
                field for field in target_fields
                if isinstance(field, str) and field in allowed_target_fields
            ]

        if next_action in {"llm_extraction_fallback", "retry_retrieval"} and not target_fields:
            target_fields = self._dedupe_fields(missing_fields + low_confidence_fields)

        failure_mode = decision.get("failure_mode")
        if not isinstance(failure_mode, str) or not failure_mode:
            failure_mode = fallback_decision.get("failure_mode", "unknown")

        reason = decision.get("reason")
        if not isinstance(reason, str) or not reason:
            reason = fallback_decision.get("reason", "No reason provided by Ollama.")

        return {
            "next_action": next_action,
            "target_fields": target_fields,
            "failure_mode": failure_mode,
            "reason": reason,
            "decision_source": "ollama",
            "model": settings.llm_model,
        }

    def _get_field_map(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Supports multiple result shapes:
        1. {"extracted_fields": {...}}
        2. {"fields": {...}}
        3. {"scored_fields": {...}}
        4. Direct field map: {"principal_amount": {"value": ...}}
        """

        if not result:
            return {}

        if isinstance(result.get("extracted_fields"), dict):
            return result["extracted_fields"]

        if isinstance(result.get("fields"), dict):
            return result["fields"]

        if isinstance(result.get("scored_fields"), dict):
            return result["scored_fields"]

        field_like_items = {}

        for key, value in result.items():
            if isinstance(value, dict) and (
                "value" in value
                or "final_confidence" in value
                or "extractor_confidence" in value
                or "confidence" in value
            ):
                field_like_items[key] = value

        return field_like_items

    def _get_missing_fields(self, extraction_result: Dict[str, Any]) -> List[str]:
        field_map = self._get_field_map(extraction_result)
        missing_fields = []

        for field_name, field_info in field_map.items():
            if isinstance(field_info, dict):
                value = field_info.get("value")
            else:
                value = field_info

            if value is None or value == "" or value == "N/A":
                missing_fields.append(field_name)

        return missing_fields

    def _get_low_confidence_fields(self, confidence_result: Dict[str, Any]) -> List[str]:
        field_map = self._get_field_map(confidence_result)
        low_confidence_fields = []

        for field_name, field_info in field_map.items():
            if not isinstance(field_info, dict):
                continue

            confidence = (
                field_info.get("final_confidence")
                or field_info.get("confidence")
                or field_info.get("extractor_confidence")
                or 0.0
            )

            if confidence < 0.75:
                low_confidence_fields.append(field_name)

        return low_confidence_fields

    def _dedupe_fields(self, fields: List[str]) -> List[str]:
        seen = set()
        deduped = []

        for field in fields:
            if field in seen:
                continue

            seen.add(field)
            deduped.append(field)

        return deduped