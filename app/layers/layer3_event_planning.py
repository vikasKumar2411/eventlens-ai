from typing import Any, Dict


DEBT_OR_FINANCING_FIELDS = [
    "borrower_or_issuer",
    "debt_type",
    "principal_amount",
    "maturity_date",
    "interest_rate",
    "use_of_proceeds",
    "lender_or_underwriter",
    "collateral_or_guarantee",
]


DEBT_OR_FINANCING_QUERIES = {
    "borrower_or_issuer": (
        "borrower issuer registrant company obligor debtor credit agreement financing"
    ),
    "debt_type": (
        "debt type notes loan credit facility term loan revolving facility senior notes convertible notes"
    ),
    "principal_amount": (
        "aggregate principal amount loan amount note amount financing amount credit facility amount"
    ),
    "maturity_date": (
        "maturity date due date matures maturity notes loan credit facility"
    ),
    "interest_rate": (
        "interest rate coupon rate SOFR LIBOR fixed rate floating rate margin payable interest"
    ),
    "use_of_proceeds": (
        "use of proceeds proceeds will be used repay debt working capital general corporate purposes"
    ),
    "lender_or_underwriter": (
        "lender administrative agent underwriter initial purchaser bank financing party"
    ),
    "collateral_or_guarantee": (
        "collateral secured unsecured guarantee guarantor obligations security interest"
    ),
}


def build_event_plan(case_id: str, event_type: str) -> Dict[str, Any]:
    if event_type != "debt_or_financing":
        raise ValueError(f"Unsupported event_type for MVP: {event_type}")

    return {
        "case_id": case_id,
        "event_type": event_type,
        "required_fields": DEBT_OR_FINANCING_FIELDS,
        "retrieval_queries": DEBT_OR_FINANCING_QUERIES,
    }


def event_planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for event planning.

    Input:
        state with case_id and event_type

    Output:
        state with plan added
    """

    case_id = state.get("case_id")
    event_type = state.get("event_type")

    if not case_id:
        raise ValueError("Missing case_id in state")

    if not event_type:
        raise ValueError("Missing event_type in state")

    plan = build_event_plan(case_id=case_id, event_type=event_type)

    completed_steps = state.get("completed_steps", [])

    if "event_planning" not in completed_steps:
        completed_steps.append("event_planning")

    state["plan"] = plan
    state["completed_steps"] = completed_steps

    return state