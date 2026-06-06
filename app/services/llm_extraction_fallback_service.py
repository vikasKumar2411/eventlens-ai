from typing import Any, Dict, List

from app.services.llm_service import LLMService
from app.observability.tracing import get_tracer


CRITICAL_FIELDS = {
    "borrower_or_issuer",
    "debt_type",
    "principal_amount",
    "maturity_date",
    "interest_rate",
    "use_of_proceeds",
    "lender_or_underwriter",
    "collateral_or_guarantee",
}


FIELD_PRIORITY = [
    "principal_amount",
    "maturity_date",
    "debt_type",
    "lender_or_underwriter",
    "interest_rate",
    "use_of_proceeds",
    "borrower_or_issuer",
    "collateral_or_guarantee",
]


MAX_LLM_FALLBACK_FIELDS = 3
MAX_EVIDENCE_CHUNKS_PER_FIELD = 4
MAX_CHARS_PER_CHUNK = 1200


class LLMExtractionFallbackService:
    """
    LLM fallback extractor for weak/missing debt_or_financing fields.

    Important:
    - It only uses retrieved evidence chunks.
    - It only extracts target fields.
    - It returns structured JSON.
    - It does not replace high-confidence deterministic fields.
    """
    
    def __init__(self):
        self.llm_service = LLMService()

    def run_fallback(
        self,
        case_id: str,
        event_type: str,
        extraction_result: Dict[str, Any],
        confidence_result: Dict[str, Any],
        evidence_bundle: Dict[str, Any],
        judge_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "eventlens.service.llm_extraction_fallback.run"
        ) as span:
            span.set_attribute("eventlens.case_id", case_id)
            span.set_attribute("eventlens.event_type", event_type)

            target_fields = self._identify_target_fields(
                confidence_result=confidence_result,
                judge_result=judge_result,
            )

            span.set_attribute("eventlens.llm_fallback.target_field_count", len(target_fields))
            span.set_attribute("eventlens.llm_fallback.target_fields", ",".join(target_fields))

            llm_field_results = {}

            for field_name in target_fields:
                with tracer.start_as_current_span(
                    "eventlens.service.llm_extraction_fallback.field"
                ) as field_span:
                    field_span.set_attribute("eventlens.field_name", field_name)

                    evidence_chunks = self._get_fallback_evidence_chunks(
                        field_name=field_name,
                        evidence_bundle=evidence_bundle,
                    )

                    field_span.set_attribute(
                        "eventlens.evidence_chunk_count",
                        len(evidence_chunks),
                    )

                    if not evidence_chunks:
                        llm_field_results[field_name] = {
                            "field_name": field_name,
                            "value": None,
                            "confidence": 0.0,
                            "evidence_quote": None,
                            "reason": "No evidence chunks available for LLM fallback.",
                            "extraction_method": "llm_fallback",
                        }
                        field_span.set_attribute("eventlens.llm.value_found", False)
                        continue

                    prompt = self._build_field_prompt(
                        field_name=field_name,
                        event_type=event_type,
                        evidence_chunks=evidence_chunks,
                    )

                    field_span.set_attribute("llm.prompt_chars", len(prompt))

                    llm_result = self.llm_service.generate_json(prompt)

                    normalized = self._normalize_llm_result(
                        field_name=field_name,
                        llm_result=llm_result,
                    )

                    field_span.set_attribute(
                        "eventlens.llm.value_found",
                        bool(normalized.get("value")),
                    )
                    field_span.set_attribute(
                        "eventlens.llm.confidence",
                        float(normalized.get("confidence") or 0.0),
                    )

                    llm_field_results[field_name] = normalized

            updated_extraction_result = self._merge_llm_results(
                extraction_result=extraction_result,
                llm_field_results=llm_field_results,
            )

            return {
                "case_id": case_id,
                "event_type": event_type,
                "target_fields": target_fields,
                "llm_field_results": llm_field_results,
                "updated_extraction_result": updated_extraction_result,
                "llm_fallback_applied": True,
            }

    def _identify_target_fields(
        self,
        confidence_result: Dict[str, Any],
        judge_result: Dict[str, Any],
    ) -> List[str]:
        target_fields = set()

        scored_fields = confidence_result.get("scored_fields", {})

        for field_name, field in scored_fields.items():
            if field_name not in CRITICAL_FIELDS:
                continue

            value = field.get("value")
            final_confidence = float(field.get("final_confidence") or 0.0)
            quality_guard_status = field.get("quality_guard_status")

            if not value:
                target_fields.add(field_name)

            if final_confidence < 0.70:
                target_fields.add(field_name)

            if quality_guard_status == "reject":
                target_fields.add(field_name)

        for check in judge_result.get("checks", []):
            details = check.get("details", {})

            for field_name in details.get("missing_fields", []):
                if field_name in CRITICAL_FIELDS:
                    target_fields.add(field_name)

            for item in details.get("low_confidence_fields", []):
                field_name = item.get("field_name")
                if field_name in CRITICAL_FIELDS:
                    target_fields.add(field_name)

            for field_name in details.get("fields_without_evidence", []):
                if field_name in CRITICAL_FIELDS:
                    target_fields.add(field_name)

            for field_name in details.get("unsupported_fields", []):
                if field_name in CRITICAL_FIELDS:
                    target_fields.add(field_name)

        ordered_fields = [
            field_name for field_name in FIELD_PRIORITY
            if field_name in target_fields
        ]

        return ordered_fields[:MAX_LLM_FALLBACK_FIELDS]

    def _get_fallback_evidence_chunks(
        self,
        field_name: str,
        evidence_bundle: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        candidate_fields = [
            field_name,
            "debt_type",
            "principal_amount",
            "maturity_date",
            "interest_rate",
            "lender_or_underwriter",
        ]

        seen_ids = set()
        chunks = []

        for candidate_field in candidate_fields:
            field_evidence = evidence_bundle.get(candidate_field, {})
            evidence_chunks = field_evidence.get("evidence_chunks", [])

            for chunk in evidence_chunks:
                chunk_id = (
                    chunk.get("chunk_id")
                    or chunk.get("id")
                    or chunk.get("point_id")
                    or chunk.get("chunk_text")
                )

                if chunk_id in seen_ids:
                    continue

                seen_ids.add(chunk_id)
                chunks.append(chunk)

        return chunks[:MAX_EVIDENCE_CHUNKS_PER_FIELD]

    def _build_field_prompt(
        self,
        field_name: str,
        event_type: str,
        evidence_chunks: List[Dict[str, Any]],
    ) -> str:
        evidence_text = self._format_evidence_chunks(evidence_chunks)
        field_instruction = self._field_instruction(field_name)

        return f"""
You are an SEC 8-K debt financing extraction agent.

Task:
Extract one field from the provided evidence only.

Event type:
{event_type}

Field to extract:
{field_name}

Field instruction:
{field_instruction}

Rules:
1. Use only the evidence provided below.
2. Do not use outside knowledge.
3. If the field is not clearly present, return value as null.
4. Return only valid JSON.
5. The evidence_quote must be a short exact quote or close excerpt from the evidence.
6. Confidence must be between 0.0 and 1.0.
7. Do not guess.

Evidence:
{evidence_text}

Return JSON exactly with this structure:
{{
  "field_name": "{field_name}",
  "value": null,
  "confidence": 0.0,
  "evidence_quote": null,
  "reason": ""
}}
"""

    def _field_instruction(self, field_name: str) -> str:
        instructions = {
            "borrower_or_issuer": (
                "Extract the borrower, issuer, registrant, obligor, or company "
                "that entered into the debt/financing transaction."
            ),
            "debt_type": (
                "Extract the type of debt or financing instrument, such as credit "
                "facility, term loan, revolving facility, senior notes, convertible "
                "notes, promissory note, or loan agreement."
            ),
            "principal_amount": (
                "Extract the principal amount, commitment amount, facility amount, "
                "aggregate principal amount, or loan amount. Prefer amounts near "
                "phrases like 'principal amount', 'commitment', 'facility', 'notes', "
                "or 'loan'."
            ),
            "maturity_date": (
                "Extract the maturity date, due date, stated maturity, final maturity, "
                "or date when the loan/facility/notes mature."
            ),
            "interest_rate": (
                "Extract the interest rate, coupon rate, applicable rate, SOFR spread, "
                "base rate, margin, or pricing terms."
            ),
            "use_of_proceeds": (
                "Extract how proceeds will be used, such as repay debt, refinance, "
                "working capital, general corporate purposes, acquisition, fees, or expenses."
            ),
            "lender_or_underwriter": (
                "Extract the lender, administrative agent, underwriter, initial purchaser, "
                "bank, arranger, or financing party."
            ),
            "collateral_or_guarantee": (
                "Extract collateral, security, guarantee, guarantor, secured/unsecured "
                "status, or loan party guarantee information."
            ),
        }

        return instructions.get(
            field_name,
            "Extract the requested field from the debt financing evidence.",
        )

    def _format_evidence_chunks(
        self,
        evidence_chunks: List[Dict[str, Any]],
    ) -> str:
        formatted_chunks = []

        for index, chunk in enumerate(
            evidence_chunks[:MAX_EVIDENCE_CHUNKS_PER_FIELD],
            start=1,
        ):
            chunk_id = (
                chunk.get("chunk_id")
                or chunk.get("id")
                or chunk.get("point_id")
                or f"chunk_{index}"
            )

            text = (
                chunk.get("text")
                or chunk.get("chunk_text")
                or chunk.get("content")
                or ""
            )

            if not text:
                payload = chunk.get("payload") or {}
                text = (
                    payload.get("text")
                    or payload.get("chunk_text")
                    or payload.get("content")
                    or ""
                )

            text = str(text).strip()

            if len(text) > MAX_CHARS_PER_CHUNK:
                text = text[:MAX_CHARS_PER_CHUNK]

            formatted_chunks.append(
                f"[Chunk {index} | id={chunk_id}]\n{text}"
            )

        return "\n\n".join(formatted_chunks)

    def _normalize_llm_result(
        self,
        field_name: str,
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if llm_result.get("error"):
            return {
                "field_name": field_name,
                "value": None,
                "confidence": 0.0,
                "evidence_quote": None,
                "reason": f"LLM response error: {llm_result.get('error')}",
                "raw_response": llm_result.get("raw_response"),
                "extraction_method": "llm_fallback",
            }

        value = llm_result.get("value")
        confidence = llm_result.get("confidence", 0.0)

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(confidence, 1.0))

        return {
            "field_name": field_name,
            "value": value,
            "confidence": confidence,
            "evidence_quote": llm_result.get("evidence_quote"),
            "reason": llm_result.get("reason"),
            "extraction_method": "llm_fallback",
        }

    def _merge_llm_results(
        self,
        extraction_result: Dict[str, Any],
        llm_field_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        updated = dict(extraction_result)
        extracted_fields = dict(updated.get("extracted_fields", {}))

        for field_name, llm_field in llm_field_results.items():
            current_field = dict(extracted_fields.get(field_name, {}))

            llm_value = llm_field.get("value")
            llm_confidence = float(llm_field.get("confidence") or 0.0)

            current_value = current_field.get("value")
            current_confidence = float(
                current_field.get("confidence")
                or current_field.get("extractor_confidence")
                or 0.0
            )

            quality_guard_status = current_field.get("quality_guard_status")

            should_replace = (
                llm_value is not None
                and str(llm_value).strip() != ""
                and llm_confidence >= 0.60
                and (
                    not current_value
                    or current_confidence < 0.70
                    or quality_guard_status == "reject"
                )
            )

            if should_replace:
                current_field.update(
                    {
                        "field_name": field_name,
                        "value": llm_value,
                        "confidence": llm_confidence,
                        "extractor_confidence": llm_confidence,
                        "extraction_method": "llm_fallback",
                        "evidence_text": [llm_field.get("evidence_quote")]
                        if llm_field.get("evidence_quote")
                        else current_field.get("evidence_text", []),
                        "evidence_quote": llm_field.get("evidence_quote"),
                        "llm_reason": llm_field.get("reason"),
                        "llm_fallback_attempted": True,
                        "llm_fallback_result": llm_field,
                        "quality_guard_status": "pass",
                        "quality_guard_reason": "Accepted from LLM fallback.",
                    }
                )
            else:
                current_field["llm_fallback_attempted"] = True
                current_field["llm_fallback_result"] = llm_field

            extracted_fields[field_name] = current_field

        updated["extracted_fields"] = extracted_fields
        updated["llm_fallback_applied"] = True

        return updated