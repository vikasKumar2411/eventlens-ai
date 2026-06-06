import re
from typing import Any, Dict, List, Optional


MISSING_VALUES = {None, "", "unknown", "not_found", "N/A"}


AMOUNT_CONTEXT_TERMS = [
    "principal amount",
    "aggregate principal",
    "aggregate principal amount",
    "outstanding principal",
    "notes outstanding",
    "principal amount of the notes",
    "facility amount",
    "commitment",
    "loan amount",
    "borrowings",
    "term loan",
    "credit facility",
]


DOLLAR_AMOUNT_PATTERN = re.compile(
    r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?\s?"
    r"(?:million|billion|thousand|m|bn)?",
    re.IGNORECASE,
)


WRITTEN_AMOUNT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?"
    r"(?:million|billion|thousand)\b",
    re.IGNORECASE,
)


class AmountRecoveryService:
    """
    Specialist recovery service for principal_amount.

    Conservative rules:
    - Only recover explicit amounts.
    - Prefer amounts near debt/principal/facility/notes context.
    - Do not infer or calculate.
    - Return None when evidence is not strong enough.
    """

    def recover(
        self,
        evidence_bundle: Dict[str, Any],
        confidence_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        principal_bundle = (evidence_bundle or {}).get("principal_amount") or {}
        chunks = principal_bundle.get("evidence_chunks") or []

        candidates = []

        for chunk in chunks:
            text = chunk.get("chunk_text") or chunk.get("text") or ""
            if not text:
                continue

            candidates.extend(
                self._extract_candidates_from_text(
                    text=text,
                    chunk=chunk,
                )
            )

        ranked_candidates = self._rank_candidates(candidates)

        if not ranked_candidates:
            return self._empty_result(
                reason="No explicit dollar amount tied to principal amount, notes, facility, commitment, or debt context was found."
            )

        best = ranked_candidates[0]

        if best["confidence"] < 0.72:
            return self._empty_result(
                reason=(
                    "Potential amount candidates were found, but none had enough "
                    "debt-specific context to safely recover principal_amount."
                ),
                candidates=ranked_candidates[:5],
            )

        recovered_field = {
            "value": best["amount"],
            "extraction_method": "amount_recovery",
            "extractor_confidence": best["confidence"],
            "evidence_quality_score": best["evidence_quality_score"],
            "final_confidence": min(
                0.96,
                round((best["confidence"] * 0.7) + (best["evidence_quality_score"] * 0.3), 2),
            ),
            "quality_guard_status": "pass",
            "confidence_reason": best["reason"],
            "reason": best["reason"],
            "evidence_quote": best["evidence_quote"],
            "evidence_chunk_ids": [best["chunk_id"]] if best.get("chunk_id") else [],
        }

        return {
            "recovered": True,
            "field_name": "principal_amount",
            "recovered_field": recovered_field,
            "selected_candidate": best,
            "candidates": ranked_candidates[:5],
            "reason": best["reason"],
        }

    def _extract_candidates_from_text(
        self,
        text: str,
        chunk: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        candidates = []

        matches = list(DOLLAR_AMOUNT_PATTERN.finditer(text))
        matches.extend(list(WRITTEN_AMOUNT_PATTERN.finditer(text)))

        for match in matches:
            amount = match.group(0).strip()
            start = max(0, match.start() - 250)
            end = min(len(text), match.end() + 250)
            window = text[start:end]

            context_hits = self._find_context_hits(window)
            evidence_quality_score = self._safe_float(chunk.get("score"), default=0.5)

            confidence = self._score_candidate(
                amount=amount,
                window=window,
                context_hits=context_hits,
                evidence_quality_score=evidence_quality_score,
            )

            candidates.append(
                {
                    "amount": amount,
                    "confidence": confidence,
                    "context_hits": context_hits,
                    "evidence_quality_score": evidence_quality_score,
                    "chunk_id": chunk.get("id"),
                    "section_title": chunk.get("section_title"),
                    "evidence_quote": self._clean_quote(window),
                    "reason": self._build_reason(
                        amount=amount,
                        context_hits=context_hits,
                        confidence=confidence,
                    ),
                }
            )

        return candidates

    def _find_context_hits(self, text: str) -> List[str]:
        lower_text = text.lower()
        return [term for term in AMOUNT_CONTEXT_TERMS if term in lower_text]

    def _parse_amount_to_number(self, amount: str) -> Optional[float]:
        cleaned = amount.lower().replace("$", "").replace(",", "").strip()

        multiplier = 1.0

        if "billion" in cleaned or cleaned.endswith("bn"):
            multiplier = 1_000_000_000.0
        elif "million" in cleaned or cleaned.endswith("m"):
            multiplier = 1_000_000.0
        elif "thousand" in cleaned:
            multiplier = 1_000.0

        cleaned = (
            cleaned.replace("billion", "")
            .replace("million", "")
            .replace("thousand", "")
            .replace("bn", "")
            .replace("m", "")
            .strip()
        )

        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None

    def _score_candidate(
        self,
        amount: str,
        window: str,
        context_hits: List[str],
        evidence_quality_score: float,
    ) -> float:
        score = 0.35

        numeric_amount = self._parse_amount_to_number(amount)

        if numeric_amount is not None and numeric_amount < 100_000:
            return 0.2

        if "$" in amount:
            score += 0.2

        if context_hits:
            score += min(0.3, 0.08 * len(context_hits))

        strong_terms = {
            "principal amount",
            "aggregate principal",
            "aggregate principal amount",
            "outstanding principal",
            "notes outstanding",
            "principal amount of the notes",
            "facility amount",
            "commitment",
            "loan amount",
        }

        if any(term in context_hits for term in strong_terms):
            score += 0.15

        has_strong_term = any(term in context_hits for term in strong_terms)

        if not has_strong_term:
            return 0.25

        if evidence_quality_score >= 0.5:
            score += 0.05

        return round(min(score, 0.95), 2)

    def _rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return sorted(
            candidates,
            key=lambda item: (
                item.get("confidence", 0),
                len(item.get("context_hits", [])),
                item.get("evidence_quality_score", 0),
            ),
            reverse=True,
        )

    def _build_reason(
        self,
        amount: str,
        context_hits: List[str],
        confidence: float,
    ) -> str:
        return (
            f"Recovered explicit principal amount candidate '{amount}' "
            f"from nearby debt context terms {context_hits}. "
            f"Specialist confidence={confidence}."
        )

    def _clean_quote(self, text: str) -> str:
        return " ".join(text.split())[:800]

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _empty_result(
        self,
        reason: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "recovered": False,
            "field_name": "principal_amount",
            "recovered_field": None,
            "selected_candidate": None,
            "candidates": candidates or [],
            "reason": reason,
        }