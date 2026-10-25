from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ScopeVisibility(StrEnum):
    GLOBAL = "global"
    TENANT = "tenant"


class RoutingTier(StrEnum):
    TIER_1_FAST_MCP = "tier_1_fast_mcp"
    TIER_2_REASONING = "tier_2_reasoning"


class ToolStepSpan(BaseModel):
    step_index: int = Field(ge=0)
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Any
    latency_ms: float = Field(ge=0.0)
    prompt_tokens: int = Field(ge=0)
    error: str | None = None


class EpisodicTrajectory(BaseModel):
    trajectory_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str
    task_intent: str
    user_query: str
    input_payload: dict[str, Any] = Field(default_factory=dict)
    steps: list[ToolStepSpan] = Field(default_factory=list)
    final_output: Any = None
    succeeded: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(step.prompt_tokens for step in self.steps)

    @property
    def total_latency_ms(self) -> float:
        return sum(step.latency_ms for step in self.steps)

    @property
    def tool_signature(self) -> tuple[str, ...]:
        return tuple(step.tool_name for step in self.steps)


class TemporalFact(BaseModel):
    fact_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str
    scope: ScopeVisibility = ScopeVisibility.TENANT
    subject: str
    predicate: str
    object_value: str
    valid_from: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_to: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.valid_to is None


class CompiledMCPToolRecord(BaseModel):
    tool_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: str
    task_intent: str
    tenant_id: str
    scope: ScopeVisibility = ScopeVisibility.TENANT
    source_code: str
    replaced_signature: tuple[str, ...]
    shadow_pass_rate: float = Field(ge=0.0, le=1.0)
    is_active: bool = True
    total_invocations: int = 0
    failed_invocations: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def success_rate(self) -> float:
        if self.total_invocations == 0:
            return 1.0
        return (self.total_invocations - self.failed_invocations) / self.total_invocations


class ShadowReplayResult(BaseModel):
    tool_name: str
    tasks_evaluated: int
    tasks_passed: int
    pass_rate: float
    avg_old_tokens: float
    avg_new_tokens: float
    token_reduction_pct: float
    promoted: bool


class AgentExecutionRequest(BaseModel):
    tenant_id: str
    task_intent: str
    user_query: str
    input_payload: dict[str, Any] = Field(default_factory=dict)


class AgentExecutionResponse(BaseModel):
    tenant_id: str
    task_intent: str
    routing_tier: RoutingTier
    steps_taken: int
    prompt_tokens_used: int
    output: Any
    active_facts_injected: list[str] = Field(default_factory=list)
    self_healing_triggered: bool = False
