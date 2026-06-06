from typing import Any, Dict, List, Tuple


class ConfidenceService:
    def score_extraction_result(
        self,
        extraction_result: Dict[str, Any],
        evidence_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        scored_fields = {}

        extracted_fields = extraction_result.get("extracted_fields", {})

        for field_name, field in extracted_fields.items():
            evidence_chunks = evidence_bundle.get(field_name, {}).get(
                "evidence_chunks", []
            )

            evidence_quality, evidence_reason = self._score_evidence_quality(
                field_name=field_name,
                field_value=field.get("value"),
                evidence_chunks=evidence_chunks,
            )

            extractor_confidence = float(field.get("confidence") or 0.0)

            final_confidence = self._combine_scores(
                extractor_confidence=extractor_confidence,
                evidence_quality=evidence_quality,
            )

            scored_fields[field_name] = {
                **field,
                "extractor_confidence": extractor_confidence,
                "evidence_quality_score": evidence_quality,
                "final_confidence": final_confidence,
                "confidence_reason": evidence_reason,
            }

        return {
            "case_id": extraction_result.get("case_id"),
            "event_type": extraction_result.get("event_type"),
            "scored_fields": scored_fields,
        }

    def _score_evidence_quality(
        self,
        field_name: str,
        field_value: Any,
        evidence_chunks: List[Dict[str, Any]],
    ) -> Tuple[float, str]:
        if not evidence_chunks:
            return 0.0, "No evidence chunks were retrieved for this field."

        top_scores = [
            float(chunk.get("score") or 0.0)
            for chunk in evidence_chunks[:3]
        ]

        avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

        supporting_text = "\n".join(
            (chunk.get("chunk_text") or "").lower()
            for chunk in evidence_chunks[:3]
        )

        if not field_value:
            return min(avg_top_score, 0.40), (
                "Evidence was retrieved, but no structured value was extracted."
            )

        value_text = str(field_value).lower()

        exact_support = value_text in supporting_text

        keyword_support = self._has_field_specific_support(
            field_name=field_name,
            supporting_text=supporting_text,
            value_text=value_text,
        )

        if exact_support and avg_top_score >= 0.50:
            return 0.90, "Extracted value appears directly in strong retrieved evidence."

        if keyword_support and avg_top_score >= 0.45:
            return 0.80, "Retrieved evidence contains strong field-specific support."

        if avg_top_score >= 0.50:
            return 0.65, "Retrieved evidence is semantically relevant but direct support is weaker."

        if avg_top_score >= 0.35:
            return 0.50, "Retrieved evidence has moderate relevance but should be reviewed."

        return 0.30, "Retrieved evidence is weak or only loosely related."

    def _has_field_specific_support(
        self,
        field_name: str,
        supporting_text: str,
        value_text: str,
    ) -> bool:
        field_keywords = {
            "borrower_or_issuer": [
                "borrower",
                "issuer",
                "registrant",
                "kbr",
                "company",
            ],
            "debt_type": [
                "term b",
                "loan",
                "facility",
                "credit agreement",
                "notes",
            ],
            "principal_amount": [
                "$",
                "principal amount",
                "aggregate principal",
                "loan amount",
            ],
            "maturity_date": [
                "maturity date",
                "matures",
                "facility shall be",
            ],
            "interest_rate": [
                "interest",
                "term sofr",
                "base rate",
                "applicable rate",
            ],
            "use_of_proceeds": [
                "proceeds",
                "repay",
                "refinance",
                "replace",
                "fees and expenses",
            ],
            "lender_or_underwriter": [
                "lender",
                "administrative agent",
                "bank of america",
                "underwriter",
            ],
            "collateral_or_guarantee": [
                "guarantor",
                "guarantee",
                "secured",
                "collateral",
            ],
        }

        keywords = field_keywords.get(field_name, [])

        has_keyword = any(keyword in supporting_text for keyword in keywords)

        # Partial support: at least one meaningful token from the value appears in evidence.
        value_tokens = [
            token
            for token in value_text.replace("/", " ").replace(";", " ").split()
            if len(token) >= 4
        ]

        has_value_overlap = any(token in supporting_text for token in value_tokens[:5])

        return has_keyword or has_value_overlap

    def _combine_scores(
        self,
        extractor_confidence: float,
        evidence_quality: float,
    ) -> float:
        # Weight extraction and evidence separately.
        # We trust extraction only when evidence quality is also strong.
        final = (0.60 * extractor_confidence) + (0.40 * evidence_quality)
        return round(final, 2)