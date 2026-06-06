from typing import Any, Dict


class SupervisorService:
    """
    Decides which EventLens agent/node should run next.

    Routing policy:
    - Normal extraction pipeline is deterministic.
    - Judge evaluates whether the output meets the goal.
    - If judge fails, RecoveryPlanner creates a bounded multi-step recovery plan.
    - PlanValidator validates the plan against allowed actions, fields, budgets, and stop rules.
    - PlanExecutor selects exactly one next step.
    - Specialist recovery actions run one at a time.
    - After each recovery action, the system re-runs judge evaluation.
    - The system stops safely through final_report.
    """

    def decide_next_action(self, state: Dict[str, Any]) -> str:
        plan = state.get("plan")
        retrieval_results = state.get("retrieval_results")
        evidence_bundle = state.get("evidence_bundle")
        extraction_result = state.get("extraction_result")
        confidence_result = state.get("confidence_result")
        summary_result = state.get("summary_result")
        judge_result = state.get("judge_result")
        final_report = state.get("final_report")

        completed_steps = state.get("completed_steps", [])

        # ------------------------------------------------------------------
        # 1. Normal deterministic pipeline sequencing
        # ------------------------------------------------------------------
        if not plan:
            return "event_planning"

        if not retrieval_results:
            return "retrieval"

        if not evidence_bundle:
            return "evidence_bundle"

        if not extraction_result:
            return "field_extraction"

        if "field_quality_guard" not in completed_steps:
            return "field_quality_guard"

        if not confidence_result:
            return "confidence_scoring"

        if not summary_result:
            return "event_summary"

        if not judge_result:
            return "judge_evaluation"

        # ------------------------------------------------------------------
        # 2. Stop/finalization checks
        # ------------------------------------------------------------------
        if final_report:
            return "stop"

        if self._judge_passed(state):
            return "final_report"

        if state.get("plan_stop_reason"):
            return "final_report"

        # ------------------------------------------------------------------
        # 3. Bounded recovery planning loop
        # ------------------------------------------------------------------

        # If a plan step is currently selected, route to the corresponding node.
        if state.get("current_plan_step"):
            return self._route_from_current_plan_step(state)

        # If judge failed and no recovery plan exists, create a plan.
        if self._judge_failed(state) and not state.get("recovery_plan"):
            return "recovery_planner"

        # If a recovery plan exists but has not been validated, validate it.
        if state.get("recovery_plan") and not state.get("validated_recovery_plan"):
            return "recovery_plan_validator"

        # If validated plan exists, select the next step or stop safely.
        if state.get("validated_recovery_plan"):
            recovery_step_count = state.get("recovery_step_count", 0)
            max_recovery_steps = state.get("max_recovery_steps", 5)

            if self._all_plan_steps_finished(state):
                state["plan_stop_reason"] = "validated_plan_completed"
                return "final_report"

            if recovery_step_count >= max_recovery_steps:
                return "final_report"

            return "recovery_plan_executor"

        # ------------------------------------------------------------------
        # 4. Backward-compatible fallback to older recovery flow
        # ------------------------------------------------------------------
        # This block keeps the previous LLM recovery supervisor path available
        # if the new planning state is absent or disabled.
        if self._judge_failed(state):
            return self._legacy_recovery_route(state)

        # ------------------------------------------------------------------
        # 5. Default finalization
        # ------------------------------------------------------------------
        return "final_report"

    # ----------------------------------------------------------------------
    # Planner-aware helper methods
    # ----------------------------------------------------------------------

    def _all_plan_steps_finished(self, state: Dict[str, Any]) -> bool:
        validated_plan = state.get("validated_recovery_plan") or {}
        steps = validated_plan.get("steps") or []

        if not steps:
            return False

        completed = state.get("completed_plan_steps") or []
        failed = state.get("failed_plan_steps") or []

        seen_steps = completed + failed

        for step in steps:
            step_action = step.get("action")
            step_fields = step.get("target_fields") or []

            already_seen = any(
                seen_step.get("action") == step_action
                and (seen_step.get("target_fields") or []) == step_fields
                for seen_step in seen_steps
            )

            if not already_seen:
                return False

        return True
        

    def _judge_failed(self, state: Dict[str, Any]) -> bool:
        judge_result = state.get("judge_result") or {}

        status = (
            judge_result.get("overall_status")
            or judge_result.get("status")
            or "unknown"
        )

        return (
            status in {"fail", "failed", "needs_recovery"}
            or judge_result.get("should_recover") is True
        )

    def _judge_passed(self, state: Dict[str, Any]) -> bool:
        judge_result = state.get("judge_result") or {}

        status = (
            judge_result.get("overall_status")
            or judge_result.get("status")
            or "unknown"
        )

        should_recover = judge_result.get("should_recover")

        return (
            status in {"pass", "passed", "success"}
            or should_recover is False
        )

    def _route_from_current_plan_step(self, state: Dict[str, Any]) -> str:
        current_step = state.get("current_plan_step") or {}
        action = current_step.get("action")

        if action == "llm_extraction_fallback":
            return "llm_extraction_fallback"

        if action == "evidence_gap_analysis":
            return "evidence_gap_analyzer"

        if action == "amount_recovery":
            return "amount_recovery"

        if action == "retry_retrieval":
            return "retrieval"

        if action == "rerun_judge":
            return "judge_evaluation"

        return "final_report"

    # ----------------------------------------------------------------------
    # Legacy recovery support
    # ----------------------------------------------------------------------

    def _legacy_recovery_route(self, state: Dict[str, Any]) -> str:
        """
        Keeps the older recovery behavior available as a safety fallback.

        Long term, this can be removed once RecoveryPlanner/Validator/Executor
        is fully stable.
        """

        final_report = state.get("final_report")
        completed_steps = state.get("completed_steps", [])

        recovery_attempts = state.get("recovery_attempts", 0)
        max_recovery_attempts = state.get("max_recovery_attempts", 1)

        llm_fallback_attempts = state.get("llm_fallback_attempts", 0)
        max_llm_fallback_attempts = state.get("max_llm_fallback_attempts", 1)

        llm_recovery_decision = state.get("llm_recovery_decision") or {}
        llm_recovery_decision_count = state.get("llm_recovery_decision_count", 0)
        max_llm_recovery_decisions = state.get("max_llm_recovery_decisions", 2)

        evidence_gap_analysis = state.get("evidence_gap_analysis") or {}
        amount_recovery_result = state.get("amount_recovery_result") or {}

        next_action = llm_recovery_decision.get("next_action")

        if (
            not llm_recovery_decision
            and llm_recovery_decision_count < max_llm_recovery_decisions
        ):
            return "llm_recovery_supervisor"

        if next_action == "llm_extraction_fallback":
            if llm_fallback_attempts < max_llm_fallback_attempts:
                return "llm_extraction_fallback"

            if not evidence_gap_analysis:
                return "evidence_gap_analyzer"

            if self._should_run_amount_recovery(
                state=state,
                evidence_gap_analysis=evidence_gap_analysis,
                amount_recovery_result=amount_recovery_result,
            ):
                return "amount_recovery"

            if llm_recovery_decision_count < max_llm_recovery_decisions:
                self._clear_recovery_decision(state)
                return "llm_recovery_supervisor"

            if not final_report:
                return "final_report"

        if next_action == "retry_retrieval":
            if not evidence_gap_analysis:
                return "evidence_gap_analyzer"

            if self._should_run_amount_recovery(
                state=state,
                evidence_gap_analysis=evidence_gap_analysis,
                amount_recovery_result=amount_recovery_result,
            ):
                return "amount_recovery"

            if recovery_attempts < max_recovery_attempts:
                return "recovery_decision"

            if llm_recovery_decision_count < max_llm_recovery_decisions:
                self._clear_recovery_decision(state)
                return "llm_recovery_supervisor"

            if not final_report:
                return "final_report"

        if next_action == "evidence_gap_analyzer":
            if not evidence_gap_analysis:
                return "evidence_gap_analyzer"

            if self._should_run_amount_recovery(
                state=state,
                evidence_gap_analysis=evidence_gap_analysis,
                amount_recovery_result=amount_recovery_result,
            ):
                return "amount_recovery"

            if llm_recovery_decision_count < max_llm_recovery_decisions:
                self._clear_recovery_decision(state)
                return "llm_recovery_supervisor"

            if not final_report:
                return "final_report"

        if next_action == "amount_recovery":
            if self._should_run_amount_recovery(
                state=state,
                evidence_gap_analysis=evidence_gap_analysis,
                amount_recovery_result=amount_recovery_result,
            ):
                return "amount_recovery"

            if not final_report:
                return "final_report"

        if next_action == "final_report":
            if not final_report:
                return "final_report"

        if (
            llm_fallback_attempts >= max_llm_fallback_attempts
            and not evidence_gap_analysis
        ):
            return "evidence_gap_analyzer"

        if self._should_run_amount_recovery(
            state=state,
            evidence_gap_analysis=evidence_gap_analysis,
            amount_recovery_result=amount_recovery_result,
        ):
            return "amount_recovery"

        if llm_recovery_decision_count < max_llm_recovery_decisions:
            self._clear_recovery_decision(state)
            return "llm_recovery_supervisor"

        return "final_report"

    def _should_run_amount_recovery(
        self,
        state: Dict[str, Any],
        evidence_gap_analysis: Dict[str, Any],
        amount_recovery_result: Dict[str, Any],
    ) -> bool:
        """
        Runs amount recovery only when:
        - evidence gap analysis exists
        - it recommends amount_recovery for principal_amount
        - amount_recovery has not already run
        """

        if amount_recovery_result:
            return False

        if not evidence_gap_analysis:
            return False

        completed_steps = state.get("completed_steps", [])

        if "amount_recovery" in completed_steps:
            return False

        recommended_actions = evidence_gap_analysis.get("recommended_actions") or []

        return any(
            action.get("field_name") == "principal_amount"
            and action.get("recommended_action") == "amount_recovery"
            for action in recommended_actions
        )

    def _clear_recovery_decision(self, state: Dict[str, Any]) -> None:
        """
        Clears the previous recovery decision so the LLM Recovery Supervisor
        can make a fresh decision after a recovery action has been exhausted.
        """

        state["llm_recovery_decision"] = {}
        state["recovery_mode"] = ""
        state["target_recovery_fields"] = []
        state["failure_mode"] = ""