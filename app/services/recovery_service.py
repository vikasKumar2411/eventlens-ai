from copy import deepcopy
from typing import Any, Dict, List


class RecoveryService:
    def decide_and_rewrite(
        self,
        plan: Dict[str, Any],
        judge_result: Dict[str, Any],
        confidence_result: Dict[str, Any],
        recovery_attempts: int,
        max_recovery_attempts: int,
    ) -> Dict[str, Any]:
        should_recover = bool(judge_result.get("should_recover"))

        if not should_recover:
            return {
                "should_recover": False,
                "recovery_reason": "Judge passed or only produced non-recovery warnings.",
                "fields_to_recover": [],
                "recovery_actions": [],
                "updated_plan": plan,
                "recovery_attempts": recovery_attempts,
                "max_recovery_attempts": max_recovery_attempts,
            }

        if recovery_attempts >= max_recovery_attempts:
            return {
                "should_recover": False,
                "recovery_reason": "Maximum recovery attempts reached.",
                "fields_to_recover": [],
                "recovery_actions": [],
                "updated_plan": plan,
                "recovery_attempts": recovery_attempts,
                "max_recovery_attempts": max_recovery_attempts,
            }

        fields_to_recover = self._identify_fields_to_recover(
            judge_result=judge_result,
            confidence_result=confidence_result,
        )

        if not fields_to_recover:
            return {
                "should_recover": False,
                "recovery_reason": "Judge requested recovery, but no recoverable fields were identified.",
                "fields_to_recover": [],
                "recovery_actions": [],
                "updated_plan": plan,
                "recovery_attempts": recovery_attempts,
                "max_recovery_attempts": max_recovery_attempts,
            }

        updated_plan = self._rewrite_plan_queries(
            plan=plan,
            fields_to_recover=fields_to_recover,
        )

        recovery_actions = [
            {
                "action": "rewrite_retrieval_query",
                "field_name": field_name,
                "old_query": plan.get("retrieval_queries", {}).get(field_name),
                "new_query": updated_plan.get("retrieval_queries", {}).get(field_name),
            }
            for field_name in fields_to_recover
        ]

        return {
            "should_recover": True,
            "recovery_reason": "Judge failed and recoverable fields were identified.",
            "fields_to_recover": fields_to_recover,
            "recovery_actions": recovery_actions,
            "updated_plan": updated_plan,
            "recovery_attempts": recovery_attempts + 1,
            "max_recovery_attempts": max_recovery_attempts,
        }

    def _identify_fields_to_recover(
        self,
        judge_result: Dict[str, Any],
        confidence_result: Dict[str, Any],
    ) -> List[str]:
        fields = set()

        for check in judge_result.get("checks", []):
            details = check.get("details", {})

            for field_name in details.get("missing_fields", []):
                fields.add(field_name)

            for item in details.get("low_confidence_fields", []):
                field_name = item.get("field_name")
                if field_name:
                    fields.add(field_name)

            for field_name in details.get("fields_without_evidence", []):
                fields.add(field_name)

            # If summary grounding fails, recover unsupported fields too.
            for field_name in details.get("unsupported_fields", []):
                fields.add(field_name)

        scored_fields = confidence_result.get("scored_fields", {})

        for field_name, field in scored_fields.items():
            value = field.get("value")
            final_confidence = float(field.get("final_confidence") or 0.0)

            if not value or final_confidence < 0.70:
                fields.add(field_name)

        return sorted(fields)

    def _rewrite_plan_queries(
        self,
        plan: Dict[str, Any],
        fields_to_recover: List[str],
    ) -> Dict[str, Any]:
        updated_plan = deepcopy(plan)
        retrieval_queries = updated_plan.get("retrieval_queries", {})

        recovery_boosters = {
            "borrower_or_issuer": (
                " named borrower issuer obligor loan party registrant exact company name"
            ),
            "debt_type": (
                " exact facility name term loan notes credit agreement amendment financing instrument"
            ),
            "principal_amount": (
                " exact dollar amount outstanding principal amount aggregate principal amount commitment"
            ),
            "maturity_date": (
                " exact maturity date matures due date final repayment installment facility maturity"
            ),
            "interest_rate": (
                " applicable rate interest margin SOFR base rate coupon pricing spread"
            ),
            "use_of_proceeds": (
                " proceeds used to repay refinance replace prepay fees expenses working capital"
            ),
            "lender_or_underwriter": (
                " administrative agent lender lenders underwriter initial purchaser bank party"
            ),
            "collateral_or_guarantee": (
                " guarantor guarantee collateral secured obligations security agreement loan parties"
            ),
        }

        for field_name in fields_to_recover:
            old_query = retrieval_queries.get(field_name, field_name)
            booster = recovery_boosters.get(field_name, " exact supporting evidence")
            retrieval_queries[field_name] = f"{old_query} {booster}".strip()

        updated_plan["retrieval_queries"] = retrieval_queries

        return updated_plan