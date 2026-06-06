from typing import Any, Dict, List, Optional, TypedDict


class EventLensState(TypedDict, total=False):
    case_id: str
    event_type: str
    goal: str
    next_action: Optional[str]
    task_status: str
    completed_steps: List[str]
    agent_trace: List[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]

    plan: Dict[str, Any]
    retrieval_results: Dict[str, Any]
    evidence_bundle: Dict[str, Any]
    extraction_result: Dict[str, Any]
    confidence_result: Dict[str, Any]
    summary_result: Dict[str, Any]
    judge_result: Dict[str, Any]

    recovery_attempts: int
    max_recovery_attempts: int
    recovery_history: List[Dict[str, Any]]
    recovery_result: Dict[str, Any]

    llm_fallback_attempted: bool
    llm_fallback_attempts: int
    max_llm_fallback_attempts: int
    llm_fallback_fields: List[str]
    llm_fallback_result: Dict[str, Any]

    llm_recovery_decision: Dict[str, Any]
    llm_recovery_decision_attempted: bool
    llm_recovery_decision_count: int
    max_llm_recovery_decisions: int
    recovery_mode: str
    target_recovery_fields: List[str]
    failure_mode: str
    autonomy_decision_history: List[Dict[str, Any]]

    preserved_recovered_fields: Dict[str, Any]

    evidence_gap_analysis: Dict[str, Any]
    amount_recovery_result: Dict[str, Any]

    post_llm_fallback_judge_result: Dict[str, Any]

    # Bounded recovery planning state
    recovery_goal: Optional[str]
    recovery_plan: Dict[str, Any]
    validated_recovery_plan: Dict[str, Any]
    current_plan_step: Optional[Dict[str, Any]]
    completed_plan_steps: List[Dict[str, Any]]
    failed_plan_steps: List[Dict[str, Any]]
    recovery_step_count: int
    max_recovery_steps: int
    plan_stop_reason: Optional[str]
    previous_judge_score: Optional[float]
    current_judge_score: Optional[float]

    # Measurable recovery metrics
    pre_recovery_judge_result: Dict[str, Any]
    post_recovery_judge_result: Dict[str, Any]
    judge_score_history: List[Dict[str, Any]]
    field_improvement_history: List[Dict[str, Any]]

    errors: List[str]