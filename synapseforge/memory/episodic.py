from __future__ import annotations

from synapseforge.models import EpisodicTrajectory


class EpisodicTrajectoryStore:
    """Append-only episodic store for tracking live execution trajectories."""

    def __init__(self) -> None:
        self._trajectories: list[EpisodicTrajectory] = []

    def log_trajectory(self, trajectory: EpisodicTrajectory) -> str:
        self._trajectories.append(trajectory)
        return trajectory.trajectory_id

    def get_successful_by_intent(
        self,
        tenant_id: str,
        task_intent: str,
        limit: int = 25,
    ) -> list[EpisodicTrajectory]:
        matches = [
            t
            for t in self._trajectories
            if t.tenant_id == tenant_id and t.task_intent == task_intent and t.succeeded
        ]
        return matches[-limit:]

    def list_all(self, tenant_id: str | None = None) -> list[EpisodicTrajectory]:
        if tenant_id is None:
            return list(self._trajectories)
        return [t for t in self._trajectories if t.tenant_id == tenant_id]
