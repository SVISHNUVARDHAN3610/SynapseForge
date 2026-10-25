from __future__ import annotations

from pydantic import BaseModel

from synapseforge.models import EpisodicTrajectory


class TelemetrySummary(BaseModel):
    baseline_avg_steps: float
    current_avg_steps: float
    step_compression_ratio: float
    baseline_avg_tokens: float
    current_avg_tokens: float
    token_savings_pct: float


def compute_telemetry_summary(
    baseline_trajectories: list[EpisodicTrajectory],
    recent_trajectories: list[EpisodicTrajectory],
) -> TelemetrySummary:
    if not baseline_trajectories or not recent_trajectories:
        return TelemetrySummary(
            baseline_avg_steps=0.0,
            current_avg_steps=0.0,
            step_compression_ratio=1.0,
            baseline_avg_tokens=0.0,
            current_avg_tokens=0.0,
            token_savings_pct=0.0,
        )

    base_steps = sum(t.step_count for t in baseline_trajectories) / len(baseline_trajectories)
    curr_steps = sum(t.step_count for t in recent_trajectories) / len(recent_trajectories)

    base_tokens = sum(t.total_prompt_tokens for t in baseline_trajectories) / len(
        baseline_trajectories
    )
    curr_tokens = sum(t.total_prompt_tokens for t in recent_trajectories) / len(
        recent_trajectories
    )

    compression_ratio = base_steps / curr_steps if curr_steps > 0 else 1.0
    token_savings_pct = (
        ((base_tokens - curr_tokens) / base_tokens) * 100.0 if base_tokens > 0 else 0.0
    )

    return TelemetrySummary(
        baseline_avg_steps=round(base_steps, 2),
        current_avg_steps=round(curr_steps, 2),
        step_compression_ratio=round(compression_ratio, 2),
        baseline_avg_tokens=round(base_tokens, 2),
        current_avg_tokens=round(curr_tokens, 2),
        token_savings_pct=round(token_savings_pct, 2),
    )
