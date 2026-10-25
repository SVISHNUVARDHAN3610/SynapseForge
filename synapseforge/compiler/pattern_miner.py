from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from synapseforge.models import EpisodicTrajectory


@dataclass(frozen=True)
class TrajectoryPatternCandidate:
    tenant_id: str
    task_intent: str
    tool_signature: tuple[str, ...]
    trajectories: list[EpisodicTrajectory]

    @property
    def average_steps(self) -> float:
        return sum(t.step_count for t in self.trajectories) / len(self.trajectories)

    @property
    def average_tokens(self) -> float:
        return sum(t.total_prompt_tokens for t in self.trajectories) / len(self.trajectories)


class TrajectoryPatternMiner:
    """Identifies recurring multi-step trajectories eligible for FastMCP tool compilation."""

    def __init__(self, min_steps: int = 4, min_occurrences: int = 5) -> None:
        self.min_steps = min_steps
        self.min_occurrences = min_occurrences

    def find_candidates(
        self,
        trajectories: list[EpisodicTrajectory],
    ) -> list[TrajectoryPatternCandidate]:
        grouped: dict[tuple[str, str, tuple[str, ...]], list[EpisodicTrajectory]] = defaultdict(list)

        for traj in trajectories:
            if not traj.succeeded or traj.step_count < self.min_steps:
                continue
            key = (traj.tenant_id, traj.task_intent, traj.tool_signature)
            grouped[key].append(traj)

        candidates: list[TrajectoryPatternCandidate] = []
        for (tenant_id, task_intent, signature), group in grouped.items():
            if len(group) >= self.min_occurrences:
                candidates.append(
                    TrajectoryPatternCandidate(
                        tenant_id=tenant_id,
                        task_intent=task_intent,
                        tool_signature=signature,
                        trajectories=group,
                    )
                )
        return candidates
