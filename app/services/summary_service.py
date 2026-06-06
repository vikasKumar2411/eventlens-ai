from typing import Any, Dict, Optional


class SummaryService:
    def generate_summary(
        self,
        confidence_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        case_id = confidence_result.get("case_id")
        event_type = confidence_result.get("event_type")
        scored_fields = confidence_result.get("scored_fields", {})

        if event_type == "debt_or_financing":
            return self._generate_debt_or_financing_summary(
                case_id=case_id,
                event_type=event_type,
                scored_fields=scored_fields,
            )

        return {
            "case_id": case_id,
            "event_type": event_type,
            "summary": None,
            "summary_confidence": 0.0,
            "summary_method": "deterministic",
            "warnings": [f"Unsupported event_type for summary: {event_type}"],
        }

    def _generate_debt_or_financing_summary(
        self,
        case_id: str,
        event_type: str,
        scored_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        borrower = self._value(scored_fields, "borrower_or_issuer")
        debt_type = self._value(scored_fields, "debt_type")
        principal_amount = self._value(scored_fields, "principal_amount")
        maturity_date = self._value(scored_fields, "maturity_date")
        interest_rate = self._value(scored_fields, "interest_rate")
        use_of_proceeds = self._value(scored_fields, "use_of_proceeds")
        lender = self._value(scored_fields, "lender_or_underwriter")
        collateral = self._value(scored_fields, "collateral_or_guarantee")

        warnings = []

        if not borrower:
            warnings.append("Missing borrower_or_issuer.")
        if not debt_type:
            warnings.append("Missing debt_type.")
        if not principal_amount:
            warnings.append("Missing principal_amount.")
        if not maturity_date:
            warnings.append("Missing maturity_date.")

        summary_parts = []

        opening = self._build_opening_sentence(
            borrower=borrower,
            debt_type=debt_type,
            principal_amount=principal_amount,
            maturity_date=maturity_date,
        )
        if opening:
            summary_parts.append(opening)

        lender_sentence = self._build_lender_sentence(lender)
        if lender_sentence:
            summary_parts.append(lender_sentence)

        proceeds_sentence = self._build_proceeds_sentence(use_of_proceeds)
        if proceeds_sentence:
            summary_parts.append(proceeds_sentence)

        interest_sentence = self._build_interest_sentence(interest_rate)
        if interest_sentence:
            summary_parts.append(interest_sentence)

        collateral_sentence = self._build_collateral_sentence(collateral)
        if collateral_sentence:
            summary_parts.append(collateral_sentence)

        summary = " ".join(summary_parts) if summary_parts else None

        summary_confidence = self._calculate_summary_confidence(scored_fields)

        return {
            "case_id": case_id,
            "event_type": event_type,
            "summary": summary,
            "summary_confidence": summary_confidence,
            "summary_method": "deterministic",
            "warnings": warnings,
            "supporting_fields": {
                "borrower_or_issuer": borrower,
                "debt_type": debt_type,
                "principal_amount": principal_amount,
                "maturity_date": maturity_date,
                "interest_rate": interest_rate,
                "use_of_proceeds": use_of_proceeds,
                "lender_or_underwriter": lender,
                "collateral_or_guarantee": collateral,
            },
        }

    def _value(
        self,
        scored_fields: Dict[str, Any],
        field_name: str,
    ) -> Optional[str]:
        field = scored_fields.get(field_name, {})
        value = field.get("value")
        return value if value else None

    def _build_opening_sentence(
        self,
        borrower: Optional[str],
        debt_type: Optional[str],
        principal_amount: Optional[str],
        maturity_date: Optional[str],
    ) -> Optional[str]:
        if not borrower and not debt_type:
            return None

        sentence = ""

        if borrower:
            sentence += f"{borrower} entered into a debt financing transaction"
        else:
            sentence += "The company entered into a debt financing transaction"

        if debt_type:
            sentence += f" involving {debt_type}"

        if principal_amount:
            sentence += f" with a principal amount of {principal_amount}"

        if maturity_date:
            sentence += f" and maturity in {maturity_date}"

        sentence += "."

        return sentence

    def _build_lender_sentence(
        self,
        lender: Optional[str],
    ) -> Optional[str]:
        if not lender:
            return None

        return f"The financing involves {lender}."

    def _build_proceeds_sentence(
        self,
        use_of_proceeds: Optional[str],
    ) -> Optional[str]:
        if not use_of_proceeds:
            return None

        return f"Use of proceeds includes {use_of_proceeds}."

    def _build_interest_sentence(
        self,
        interest_rate: Optional[str],
    ) -> Optional[str]:
        if not interest_rate:
            return None

        return f"The interest terms reference {interest_rate}."

    def _build_collateral_sentence(
        self,
        collateral: Optional[str],
    ) -> Optional[str]:
        if not collateral:
            return None

        # Keep guarantor lists from making the summary too long.
        if len(collateral) > 220:
            collateral = collateral[:220].rstrip() + "..."

        return f"Guarantee or collateral information includes {collateral}."

    def _calculate_summary_confidence(
        self,
        scored_fields: Dict[str, Any],
    ) -> float:
        important_fields = [
            "borrower_or_issuer",
            "debt_type",
            "principal_amount",
            "maturity_date",
            "lender_or_underwriter",
            "use_of_proceeds",
        ]

        scores = []

        for field_name in important_fields:
            field = scored_fields.get(field_name, {})
            value = field.get("value")
            confidence = field.get("final_confidence")

            if value and confidence is not None:
                scores.append(float(confidence))

        if not scores:
            return 0.0

        return round(sum(scores) / len(scores), 2)