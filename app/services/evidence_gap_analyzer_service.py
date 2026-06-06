from typing import Any, Dict, List


MISSING_VALUES = {None, "", "unknown", "not_found", "N/A"}


FIELD_KEYWORDS = {
    "debt_type": [
        "notes",
        "credit agreement",
        "term loan",
        "revolving facility",
        "loan agreement",
        "senior notes",
        "convertible notes",
        "facility",
        "indenture",
    ],
    "principal_amount": [
        "$",
        "principal amount",
        "aggregate principal",
        "commitment",
        "facility amount",
        "loan amount",
        "notes outstanding",
        "amount outstanding",
        "borrowings",
    ],
    "maturity_date": [
        "maturity",
        "matures",
        "due date",
        "stated maturity",
    ],
    "interest_rate": [
        "interest rate",
        "sofr",
        "libor",
        "base rate",
        "coupon",
        "margin",
        "spread",
    ],
    "use_of_proceeds": [
        "use of proceeds",
        "proceeds",
        "repay",
        "refinance",
        "working capital",
        "general corporate purposes",
    ],
    "lender_or_underwriter": [
        "lender",
        "administrative agent",
        "underwriter",
        "initial purchaser",
        "bank",
        "financing party",
    ],
    "collateral_or_guarantee": [
        "collateral",
        "secured",
        "unsecured",
        "guarantee",
        "guarantor",
        "security interest",
    ],
    "borrower_or_issuer": [
        "borrower",
        "issuer",
        "registrant",
        "company",
        "obligor",
    ],
}


class EvidenceGapAnalyzerService:
    """
    Diagnoses why weak or missing fields failed.

    This is intentionally deterministic first:
    - no LLM cost
    - stable batch evaluation
    - easier to explain in interviews
    """

    def analyze(
        self,
        confidence_result: Dict[str, Any],
        evidence_bundle: Dict[str, Any],
        target_fields: List[str],
    ) -> Dict[str, Any]:
        scored_fields = (confidence_result or {}).get("scored_fields") or {}

        field_gap_analysis = {}

        for field_name in target_fields:
            field = scored_fields.get(field_name) or {}
            bundle = (evidence_bundle or {}).get(field_name) or {}
            chunks = bundle.get("evidence_chunks") or []

            value = field.get("value")
            final_confidence = field.get("final_confidence") or 0.0
            extractor_confidence = field.get("extractor_confidence") or 0.0
            evidence_quality_score = field.get("evidence_quality_score") or 0.0

            combined_text = self._combine_chunk_text(chunks)
            keyword_hits = self._find_keyword_hits(field_name, combined_text)

            failure_type = self._classify_failure(
                value=value,
                final_confidence=final_confidence,
                extractor_confidence=extractor_confidence,
                evidence_quality_score=evidence_quality_score,
                chunks=chunks,
                keyword_hits=keyword_hits,
            )

            field_gap_analysis[field_name] = {
                "field_name": field_name,
                "failure_type": failure_type,
                "reason": self._build_reason(
                    field_name=field_name,
                    failure_type=failure_type,
                    chunks=chunks,
                    keyword_hits=keyword_hits,
                    value=value,
                    final_confidence=final_confidence,
                    extractor_confidence=extractor_confidence,
                    evidence_quality_score=evidence_quality_score,
                ),
                "current_value": value,
                "final_confidence": final_confidence,
                "extractor_confidence": extractor_confidence,
                "evidence_quality_score": evidence_quality_score,
                "evidence_chunk_count": len(chunks),
                "keyword_hits": keyword_hits,
                "sample_evidence": self._sample_evidence(chunks),
            }

        return {
            "target_fields": target_fields,
            "field_gap_analysis": field_gap_analysis,
            "recommended_actions": self._recommend_actions(field_gap_analysis),
        }

    def _combine_chunk_text(self, chunks: List[Dict[str, Any]]) -> str:
        texts = []

        for chunk in chunks:
            text = chunk.get("chunk_text") or chunk.get("text") or ""
            if text:
                texts.append(text)

        return "\n".join(texts).lower()

    def _find_keyword_hits(self, field_name: str, text: str) -> List[str]:
        keywords = FIELD_KEYWORDS.get(field_name, [])
        return [keyword for keyword in keywords if keyword.lower() in text]

    def _classify_failure(
        self,
        value: Any,
        final_confidence: float,
        extractor_confidence: float,
        evidence_quality_score: float,
        chunks: List[Dict[str, Any]],
        keyword_hits: List[str],
    ) -> str:
        has_value = value not in MISSING_VALUES
        has_chunks = len(chunks) > 0
        has_field_terms = len(keyword_hits) > 0

        if not has_chunks:
            return "missing_evidence"

        if has_value and final_confidence < 0.6:
            return "unsupported_inference"

        if not has_value and has_field_terms and evidence_quality_score >= 0.35:
            return "extractor_failed_despite_evidence"

        if not has_value and has_chunks and not has_field_terms:
            return "weak_retrieval"

        if not has_value and evidence_quality_score < 0.35:
            return "weak_retrieval"

        return "missing_evidence"

    def _build_reason(
        self,
        field_name: str,
        failure_type: str,
        chunks: List[Dict[str, Any]],
        keyword_hits: List[str],
        value: Any,
        final_confidence: float,
        extractor_confidence: float,
        evidence_quality_score: float,
    ) -> str:
        if failure_type == "missing_evidence":
            return (
                f"No useful evidence was retrieved for {field_name}. "
                "The field should not be inferred without supporting text."
            )

        if failure_type == "weak_retrieval":
            return (
                f"Evidence chunks were retrieved for {field_name}, but they do not contain "
                f"strong field-specific terms. Keyword hits: {keyword_hits}."
            )

        if failure_type == "extractor_failed_despite_evidence":
            return (
                f"Evidence appears to contain field-specific terms for {field_name}, "
                f"but extraction did not produce a usable value. Keyword hits: {keyword_hits}."
            )

        if failure_type == "unsupported_inference":
            return (
                f"A value was produced for {field_name}, but confidence is weak "
                f"or grounding is insufficient. value={value}, final_confidence={final_confidence}."
            )

        if failure_type == "contradictory_evidence":
            return (
                f"Evidence for {field_name} appears inconsistent across chunks."
            )

        return (
            f"Unable to confidently classify failure for {field_name}. "
            f"extractor_confidence={extractor_confidence}, "
            f"evidence_quality_score={evidence_quality_score}."
        )

    def _sample_evidence(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        samples = []

        for chunk in chunks[:2]:
            text = chunk.get("chunk_text") or chunk.get("text") or ""

            samples.append(
                {
                    "chunk_id": chunk.get("id"),
                    "score": chunk.get("score"),
                    "section_title": chunk.get("section_title"),
                    "text_preview": text[:500],
                }
            )

        return samples

    def _recommend_actions(
        self,
        field_gap_analysis: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        actions = []

        for field_name, analysis in field_gap_analysis.items():
            failure_type = analysis.get("failure_type")

            if failure_type in {"missing_evidence", "weak_retrieval"}:
                action = "retry_retrieval"
            elif failure_type == "extractor_failed_despite_evidence":
                action = self._specialist_action_for_field(field_name)
            elif failure_type == "unsupported_inference":
                action = "lower_confidence_or_remove_value"
            else:
                action = "manual_review"

            actions.append(
                {
                    "field_name": field_name,
                    "failure_type": failure_type,
                    "recommended_action": action,
                }
            )

        return actions

    def _specialist_action_for_field(self, field_name: str) -> str:
        if field_name == "principal_amount":
            return "amount_recovery"

        if field_name == "debt_type":
            return "debt_type_recovery"

        return "llm_extraction_fallback"