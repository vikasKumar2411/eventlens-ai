import re
from typing import Any, Dict, List, Optional

from app.schemas.extraction import ExtractedField


class ExtractionService:
    def extract_for_evidence_bundle(
        self,
        case_id: str,
        event_type: str,
        evidence_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        extracted_fields = {}

        for field_name, bundle in evidence_bundle.items():
            chunks = bundle.get("evidence_chunks", [])
            extracted_fields[field_name] = self._extract_field(field_name, chunks)

        return {
            "case_id": case_id,
            "event_type": event_type,
            "extracted_fields": {
                field_name: field.model_dump()
                for field_name, field in extracted_fields.items()
            },
        }

    def _extract_field(
        self,
        field_name: str,
        chunks: List[Dict[str, Any]],
    ) -> ExtractedField:
        if not chunks:
            return ExtractedField(
                field_name=field_name,
                value=None,
                confidence=0.0,
            )

        combined_text = "\n\n".join(
            chunk.get("chunk_text") or "" for chunk in chunks
        )

        if field_name == "borrower_or_issuer":
            return self._extract_borrower_or_issuer(field_name, chunks, combined_text)

        if field_name == "debt_type":
            return self._extract_debt_type(field_name, chunks, combined_text)

        if field_name == "principal_amount":
            return self._extract_principal_amount(field_name, chunks, combined_text)

        if field_name == "maturity_date":
            return self._extract_maturity_date(field_name, chunks, combined_text)

        if field_name == "interest_rate":
            return self._extract_interest_rate(field_name, chunks, combined_text)

        if field_name == "use_of_proceeds":
            return self._extract_use_of_proceeds(field_name, chunks, combined_text)

        if field_name == "lender_or_underwriter":
            return self._extract_lender_or_underwriter(field_name, chunks, combined_text)

        if field_name == "collateral_or_guarantee":
            return self._extract_collateral_or_guarantee(field_name, chunks, combined_text)

        return self._fallback(field_name, chunks)

    def _top_evidence(
        self,
        chunks: List[Dict[str, Any]],
        max_items: int = 2,
    ) -> tuple[List[str], List[str]]:
        top_chunks = chunks[:max_items]

        ids = [str(chunk.get("id")) for chunk in top_chunks if chunk.get("id") is not None]
        texts = [
            (chunk.get("chunk_text") or "")[:600]
            for chunk in top_chunks
            if chunk.get("chunk_text")
        ]

        return ids, texts

    def _extract_borrower_or_issuer(
        self,
        field_name: str,
        chunks: List[Dict[str, Any]],
        text: str,
    ) -> ExtractedField:
        company_name = chunks[0].get("company_name")
        ids, texts = self._top_evidence(chunks)

        if company_name:
            return ExtractedField(
                field_name=field_name,
                value=company_name,
                confidence=0.95,
                evidence_chunk_ids=ids,
                evidence_text=texts,
            )

        match = re.search(
            r"among\s+([A-Z0-9&.,\-\s]+?),\s+a\s+[A-Za-z\s]+corporation",
            text,
            flags=re.IGNORECASE,
        )

        value = match.group(1).strip() if match else None

        return ExtractedField(
            field_name=field_name,
            value=value,
            confidence=0.75 if value else 0.0,
            evidence_chunk_ids=ids,
            evidence_text=texts,
        )

    def _extract_debt_type(
        self,
        field_name: str,
        chunks: List[Dict[str, Any]],
        text: str,
    ) -> ExtractedField:
        ids, texts = self._top_evidence(chunks)

        patterns = [
            r"Incremental Term B Loan",
            r"Incremental Term B Loans",
            r"Term B Facility",
            r"Term B Loans",
            r"Credit Agreement",
            r"senior notes",
            r"convertible notes",
            r"revolving credit facility",
        ]

        found = []
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                found.append(pattern)

        value = " / ".join(dict.fromkeys(found)) if found else None

        return ExtractedField(
            field_name=field_name,
            value=value,
            confidence=0.85 if value else 0.0,
            evidence_chunk_ids=ids,
            evidence_text=texts,
        )

    def _extract_principal_amount(
        self,
        field_name: str,
        chunks: List[Dict[str, Any]],
        text: str,
    ) -> ExtractedField:
        ids, texts = self._top_evidence(chunks)

        amount_patterns = [
            r"\$[\d,]+(?:\.\d+)?",
            r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\s+dollars\b",
        ]

        amounts = []
        for pattern in amount_patterns:
            amounts.extend(re.findall(pattern, text, flags=re.IGNORECASE))

        value = None
        if amounts:
            # Prefer largest dollar-looking amount.
            def amount_to_number(amount: str) -> float:
                cleaned = re.sub(r"[^\d.]", "", amount)
                try:
                    return float(cleaned)
                except ValueError:
                    return 0.0

            value = max(amounts, key=amount_to_number)

        return ExtractedField(
            field_name=field_name,
            value=value,
            confidence=0.85 if value else 0.0,
            evidence_chunk_ids=ids,
            evidence_text=texts,
        )

    def _extract_maturity_date(
        self,
        field_name: str,
        chunks: List[Dict[str, Any]],
        text: str,
    ) -> ExtractedField:
        ids, texts = self._top_evidence(chunks)

        # Priority 1: patterns where year appears directly before maturity language.
        direct_before_match = re.search(
            r"\b((?:20)\d{2})\b.{0,120}?\bMaturity Date\b",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if direct_before_match:
            value = direct_before_match.group(1)
            return ExtractedField(
                field_name=field_name,
                value=value,
                confidence=0.85,
                evidence_chunk_ids=ids,
                evidence_text=texts,
            )

        # Priority 2: patterns where maturity language appears before the year.
        direct_after_match = re.search(
            r"\bMaturity Date\b.{0,160}?\b((?:20)\d{2})\b",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if direct_after_match:
            value = direct_after_match.group(1)
            return ExtractedField(
                field_name=field_name,
                value=value,
                confidence=0.80,
                evidence_chunk_ids=ids,
                evidence_text=texts,
            )

        # Priority 3: choose the farthest future year near maturity/facility language.
        maturity_windows = re.findall(
            r".{0,120}(?:maturity|matures|facility).{0,120}",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        candidate_years = []
        for window in maturity_windows:
            years = re.findall(r"\b(20\d{2})\b", window)
            candidate_years.extend(years)

        if candidate_years:
            value = max(candidate_years)
            return ExtractedField(
                field_name=field_name,
                value=value,
                confidence=0.70,
                evidence_chunk_ids=ids,
                evidence_text=texts,
            )

        return ExtractedField(
            field_name=field_name,
            value=None,
            confidence=0.0,
            evidence_chunk_ids=ids,
            evidence_text=texts,
        )

    def _extract_interest_rate(
        self,
        field_name: str,
        chunks: List[Dict[str, Any]],
        text: str,
    ) -> ExtractedField:
        ids, texts = self._top_evidence(chunks)

        candidates = []

        if re.search(r"Term SOFR", text, flags=re.IGNORECASE):
            candidates.append("Term SOFR Loan")

        if re.search(r"Base Rate", text, flags=re.IGNORECASE):
            candidates.append("Base Rate Loan")

        rate_match = re.search(
            r"(\d+(?:\.\d+)?%)",
            text,
            flags=re.IGNORECASE,
        )

        if rate_match:
            candidates.append(rate_match.group(1))

        value = " / ".join(dict.fromkeys(candidates)) if candidates else None

        return ExtractedField(
            field_name=field_name,
            value=value,
            confidence=0.65 if value else 0.0,
            evidence_chunk_ids=ids,
            evidence_text=texts,
        )

    def _extract_use_of_proceeds(
        self,
        field_name: str,
        chunks: List[Dict[str, Any]],
        text: str,
    ) -> ExtractedField:
        ids, texts = self._top_evidence(chunks, max_items=3)

        proceeds_signals = []

        if re.search(r"repay|refinance|replace", text, flags=re.IGNORECASE):
            proceeds_signals.append("repay, refinance and/or replace existing debt")

        if re.search(r"Revolving Credit Loans", text, flags=re.IGNORECASE):
            proceeds_signals.append("optionally prepay Revolving Credit Loans")

        if re.search(r"fees and expenses", text, flags=re.IGNORECASE):
            proceeds_signals.append("pay fees and expenses")

        if re.search(r"accrued and unpaid interest", text, flags=re.IGNORECASE):
            proceeds_signals.append("pay accrued and unpaid interest")

        value = "; ".join(dict.fromkeys(proceeds_signals)) if proceeds_signals else None

        return ExtractedField(
            field_name=field_name,
            value=value,
            confidence=0.75 if value else 0.0,
            evidence_chunk_ids=ids,
            evidence_text=texts,
        )

    def _extract_lender_or_underwriter(
        self,
        field_name: str,
        chunks: List[Dict[str, Any]],
        text: str,
    ) -> ExtractedField:
        ids, texts = self._top_evidence(chunks)

        candidates = []

        boa_match = re.search(
            r"BANK OF AMERICA,\s*N\.A\.",
            text,
            flags=re.IGNORECASE,
        )
        if boa_match:
            candidates.append("Bank of America, N.A.")

        if re.search(r"Administrative Agent", text, flags=re.IGNORECASE):
            if candidates:
                candidates[0] = candidates[0] + ", as Administrative Agent"
            else:
                candidates.append("Administrative Agent")

        if re.search(r"Incremental Term B Lenders", text, flags=re.IGNORECASE):
            candidates.append("Incremental Term B Lenders")

        value = "; ".join(dict.fromkeys(candidates)) if candidates else None

        return ExtractedField(
            field_name=field_name,
            value=value,
            confidence=0.85 if value else 0.0,
            evidence_chunk_ids=ids,
            evidence_text=texts,
        )

    def _extract_collateral_or_guarantee(
        self,
        field_name: str,
        chunks: List[Dict[str, Any]],
        text: str,
    ) -> ExtractedField:
        ids, texts = self._top_evidence(chunks)

        guarantors = re.findall(
            r"([A-Z0-9&.,\-\s]+?),\s+a\s+[A-Za-z\s]+(?:company|corporation|liability company),\s+as a Guarantor",
            text,
            flags=re.IGNORECASE,
        )
        
        cleaned_guarantors = [
            " ".join(g.strip().split())
            for g in guarantors
            if len(g.strip()) > 2
        ]

        if cleaned_guarantors:
            value = "Guarantors: " + "; ".join(dict.fromkeys(cleaned_guarantors[:10]))
            confidence = 0.85
        elif re.search(r"Guarantor|Guarantee", text, flags=re.IGNORECASE):
            value = "Guarantor obligations referenced"
            confidence = 0.65
        else:
            value = None
            confidence = 0.0

        return ExtractedField(
            field_name=field_name,
            value=value,
            confidence=confidence,
            evidence_chunk_ids=ids,
            evidence_text=texts,
        )

    def _fallback(
        self,
        field_name: str,
        chunks: List[Dict[str, Any]],
    ) -> ExtractedField:
        ids, texts = self._top_evidence(chunks)

        return ExtractedField(
            field_name=field_name,
            value=None,
            confidence=0.0,
            evidence_chunk_ids=ids,
            evidence_text=texts,
        )