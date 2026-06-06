import re
from typing import Any, Dict


class FieldQualityGuardService:
    """
    Validates extracted fields before confidence scoring.

    This prevents suspicious deterministic extractions from flowing into
    summary and judge evaluation.
    """

    def validate_extraction_result(
        self,
        extraction_result: Dict[str, Any],
        evidence_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        extracted_fields = extraction_result.get("extracted_fields", {})

        guarded_fields = {}

        for field_name, field in extracted_fields.items():
            guarded_field = dict(field)

            validation = self._validate_field(
                field_name=field_name,
                field=field,
                evidence_bundle=evidence_bundle,
            )

            guarded_field["quality_guard_status"] = validation["status"]
            guarded_field["quality_guard_reason"] = validation["reason"]

            if validation["status"] == "reject":
                guarded_field["value"] = None
                guarded_field["confidence"] = min(
                    float(guarded_field.get("confidence") or 0.0),
                    0.2,
                )

            guarded_fields[field_name] = guarded_field

        return {
            **extraction_result,
            "extracted_fields": guarded_fields,
            "quality_guard_applied": True,
        }

    def _validate_field(
        self,
        field_name: str,
        field: Dict[str, Any],
        evidence_bundle: Dict[str, Any],
    ) -> Dict[str, str]:
        value = field.get("value")

        if not value:
            return {
                "status": "pass",
                "reason": "No value extracted, so no quality rejection applied.",
            }

        if field_name == "principal_amount":
            return self._validate_principal_amount(value=value, field=field)

        if field_name == "maturity_date":
            return self._validate_maturity_date(value=value, field=field)

        return {
            "status": "pass",
            "reason": "No special quality rule for this field.",
        }

    def _validate_principal_amount(
        self,
        value: Any,
        field: Dict[str, Any],
    ) -> Dict[str, str]:
        amount = self._parse_money(value)

        if amount is None:
            return {
                "status": "reject",
                "reason": "Principal amount could not be parsed as a numeric money value.",
            }

        evidence_text = " ".join(field.get("evidence_text", [])).lower()

        strong_context_terms = [
            "principal amount",
            "aggregate principal amount",
            "commitment",
            "credit facility",
            "term loan",
            "term b",
            "notes",
            "loan",
            "loans",
            "facility",
            "borrowing",
        ]

        has_strong_context = any(term in evidence_text for term in strong_context_terms)

        if amount < 1_000_000 and not has_strong_context:
            return {
                "status": "reject",
                "reason": (
                    "Principal amount is below $1,000,000 and lacks strong "
                    "debt-financing context."
                ),
            }

        if amount < 10_000 and has_strong_context:
            return {
                "status": "reject",
                "reason": (
                    "Principal amount is extremely small for debt financing, "
                    "even though some debt context exists."
                ),
            }

        return {
            "status": "pass",
            "reason": "Principal amount passed amount and context checks.",
        }

    def _validate_maturity_date(
        self,
        value: Any,
        field: Dict[str, Any],
    ) -> Dict[str, str]:
        text = str(value).strip()
        evidence_text = " ".join(field.get("evidence_text", [])).lower()

        maturity_terms = [
            "maturity date",
            "matures",
            "mature",
            "due date",
            "final maturity",
            "scheduled maturity",
        ]

        has_maturity_context = any(term in evidence_text for term in maturity_terms)

        # Accept 4-digit years like 2028 or 2031 only when maturity context exists.
        if re.fullmatch(r"20\d{2}", text):
            if has_maturity_context:
                return {
                    "status": "pass",
                    "reason": "Maturity year has supporting maturity context.",
                }

            return {
                "status": "reject",
                "reason": "Maturity year lacks maturity-specific evidence context.",
            }

        # Accept common full-date formats if maturity context exists.
        if re.search(
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|\d{1,2}/\d{1,2}/20\d{2})",
            text.lower(),
        ):
            if has_maturity_context:
                return {
                    "status": "pass",
                    "reason": "Maturity date has supporting maturity context.",
                }

        return {
            "status": "reject",
            "reason": "Maturity date does not have enough maturity-specific support.",
        }

    def _parse_money(self, value: Any):
        text = str(value)
        text = text.replace("$", "").replace(",", "").strip()

        # Remove trailing periods.
        text = text.rstrip(".")

        try:
            return float(text)
        except ValueError:
            return None