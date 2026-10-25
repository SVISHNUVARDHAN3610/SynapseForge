from __future__ import annotations

from collections.abc import Callable
from typing import Any

from synapseforge.compiler.mcp_compiler import FastMCPToolCompiler
from synapseforge.memory.episodic import EpisodicTrajectoryStore
from synapseforge.memory.temporal_graph import BiTemporalFactGraph
from synapseforge.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    CompiledMCPToolRecord,
    EpisodicTrajectory,
    RoutingTier,
    ScopeVisibility,
    ToolStepSpan,
)


class TwoSpeedSmartRouter:
    """Orchestrates Tier 1 (1-step FastMCP) and Tier 2 (LangGraph multi-step reasoning) execution."""

    def __init__(
        self,
        episodic_store: EpisodicTrajectoryStore,
        fact_graph: BiTemporalFactGraph,
        compiler: FastMCPToolCompiler,
        fallback_reasoning_fn: Callable[[AgentExecutionRequest, list[str]], tuple[Any, list[ToolStepSpan]]],
        min_tool_health_rate: float = 0.80,
    ) -> None:
        self.episodic_store = episodic_store
        self.fact_graph = fact_graph
        self.compiler = compiler
        self.fallback_reasoning_fn = fallback_reasoning_fn
        self.min_tool_health_rate = min_tool_health_rate
        self._registered_tools: dict[str, CompiledMCPToolRecord] = {}
        self.recompilation_queue: list[str] = []

    def register_compiled_tool(self, record: CompiledMCPToolRecord) -> None:
        self._registered_tools[record.tool_name] = record

    def find_matching_tool(
        self,
        tenant_id: str,
        task_intent: str,
    ) -> CompiledMCPToolRecord | None:
        for record in self._registered_tools.values():
            if not record.is_active or record.task_intent != task_intent:
                continue
            if record.scope == ScopeVisibility.GLOBAL or record.tenant_id == tenant_id:
                return record
        return None

    def execute(self, request: AgentExecutionRequest) -> AgentExecutionResponse:
        matched_tool = self.find_matching_tool(request.tenant_id, request.task_intent)
        self_healing_triggered = False

        if matched_tool is not None:
            matched_tool.total_invocations += 1
            try:
                fn = self.compiler.get_callable(matched_tool.tool_name)
                result = fn(request.input_payload)

                span = ToolStepSpan(
                    step_index=0,
                    tool_name=matched_tool.tool_name,
                    tool_input=request.input_payload,
                    tool_output=result,
                    latency_ms=12.0,
                    prompt_tokens=120,
                )
                self.episodic_store.log_trajectory(
                    EpisodicTrajectory(
                        tenant_id=request.tenant_id,
                        task_intent=request.task_intent,
                        user_query=request.user_query,
                        input_payload=request.input_payload,
                        steps=[span],
                        final_output=result,
                        succeeded=True,
                    )
                )
                return AgentExecutionResponse(
                    tenant_id=request.tenant_id,
                    task_intent=request.task_intent,
                    routing_tier=RoutingTier.TIER_1_FAST_MCP,
                    steps_taken=1,
                    prompt_tokens_used=120,
                    output=result,
                    active_facts_injected=[],
                    self_healing_triggered=False,
                )
            except Exception:
                matched_tool.failed_invocations += 1
                if matched_tool.success_rate < self.min_tool_health_rate:
                    matched_tool.is_active = False
                    self_healing_triggered = True
                    self.recompilation_queue.append(matched_tool.tool_name)

        # Tier 2: Full Multi-Step Reasoning Path + Bi-Temporal Fact Graph Retrieval
        active_facts = self.fact_graph.query_active_facts(tenant_id=request.tenant_id)
        fact_strings = [f"{f.subject}.{f.predicate}={f.object_value}" for f in active_facts]

        output, steps = self.fallback_reasoning_fn(request, fact_strings)
        traj = EpisodicTrajectory(
            tenant_id=request.tenant_id,
            task_intent=request.task_intent,
            user_query=request.user_query,
            input_payload=request.input_payload,
            steps=steps,
            final_output=output,
            succeeded=True,
        )
        self.episodic_store.log_trajectory(traj)

        return AgentExecutionResponse(
            tenant_id=request.tenant_id,
            task_intent=request.task_intent,
            routing_tier=RoutingTier.TIER_2_REASONING,
            steps_taken=traj.step_count,
            prompt_tokens_used=traj.total_prompt_tokens,
            output=output,
            active_facts_injected=fact_strings,
            self_healing_triggered=self_healing_triggered,
        )
