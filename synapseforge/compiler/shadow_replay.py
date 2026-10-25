from __future__ import annotations

from collections.abc import Callable
from typing import Any

from synapseforge.models import (
    CompiledMCPToolRecord,
    EpisodicTrajectory,
    ShadowReplayResult,
)


class LocalShadowReplayRunner:
    """Verifies a candidate 1-step FastMCP tool against 25 past trajectories locally."""

    def __init__(
        self,
        replay_sample_size: int = 25,
        required_pass_rate: float = 1.0,
        compiled_step_token_cost: int = 120,
    ) -> None:
        self.replay_sample_size = replay_sample_size
        self.required_pass_rate = required_pass_rate
        self.compiled_step_token_cost = compiled_step_token_cost

    def verify_and_promote(
        self,
        tool_record: CompiledMCPToolRecord,
        tool_callable: Callable[[dict[str, Any]], Any],
        historical_trajectories: list[EpisodicTrajectory],
    ) -> ShadowReplayResult:
        sample = historical_trajectories[-self.replay_sample_size :]
        if not sample:
            return ShadowReplayResult(
                tool_name=tool_record.tool_name,
                tasks_evaluated=0,
                tasks_passed=0,
                pass_rate=0.0,
                avg_old_tokens=0.0,
                avg_new_tokens=0.0,
                token_reduction_pct=0.0,
                promoted=False,
            )

        passed = 0
        total_old_tokens = 0

        for traj in sample:
            total_old_tokens += traj.total_prompt_tokens
            try:
                candidate_output = tool_callable(traj.input_payload)
                if candidate_output == traj.final_output:
                    passed += 1
            except Exception:
                continue

        evaluated = len(sample)
        pass_rate = passed / evaluated
        avg_old_tokens = total_old_tokens / evaluated
        avg_new_tokens = float(self.compiled_step_token_cost)
        token_reduction_pct = (
            ((avg_old_tokens - avg_new_tokens) / avg_old_tokens) * 100.0
            if avg_old_tokens > 0
            else 0.0
        )

        promoted = pass_rate >= self.required_pass_rate
        tool_record.shadow_pass_rate = pass_rate
        tool_record.is_active = promoted

        return ShadowReplayResult(
            tool_name=tool_record.tool_name,
            tasks_evaluated=evaluated,
            tasks_passed=passed,
            pass_rate=pass_rate,
            avg_old_tokens=avg_old_tokens,
            avg_new_tokens=avg_new_tokens,
            token_reduction_pct=token_reduction_pct,
            promoted=promoted,
        )
