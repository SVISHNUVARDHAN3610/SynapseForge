from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from synapseforge.compiler.mcp_compiler import FastMCPToolCompiler
from synapseforge.compiler.pattern_miner import TrajectoryPatternMiner
from synapseforge.compiler.shadow_replay import LocalShadowReplayRunner
from synapseforge.memory.context_manager import SmartContextManager
from synapseforge.memory.episodic import EpisodicTrajectoryStore
from synapseforge.memory.temporal_graph import BiTemporalFactGraph
from synapseforge.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    ShadowReplayResult,
    ToolStepSpan,
)
from synapseforge.orchestration.router import TwoSpeedSmartRouter

app = FastAPI(title="SynapseForge — Agent Memory & Auto-Tool Compiler", version="0.1.0")

episodic_store = EpisodicTrajectoryStore()
fact_graph = BiTemporalFactGraph()
context_manager = SmartContextManager()
pattern_miner = TrajectoryPatternMiner(min_steps=4, min_occurrences=5)
mcp_compiler = FastMCPToolCompiler()
shadow_runner = LocalShadowReplayRunner(replay_sample_size=25, required_pass_rate=1.0)


def default_reasoning_engine(
    request: AgentExecutionRequest,
    facts: list[str],
) -> tuple[Any, list[ToolStepSpan]]:
    step_names = [
        "fetch_auth_session",
        "fetch_crm_account",
        "fetch_billing_ledger",
        "fetch_usage_events",
        "filter_active_seats",
        "normalize_currency",
        "reconcile_discrepancies",
        "format_executive_summary",
    ]
    spans: list[ToolStepSpan] = []
    for idx, name in enumerate(step_names):
        spans.append(
            ToolStepSpan(
                step_index=idx,
                tool_name=name,
                tool_input=request.input_payload,
                tool_output={"step": name, "status": "ok"},
                latency_ms=45.0,
                prompt_tokens=220,
            )
        )
    final_output = {
        "user_id": request.input_payload.get("user_id"),
        "reconciled_mrr": request.input_payload.get("base_amount", 0) * 12,
        "status": "VERIFIED",
    }
    return final_output, spans


router_engine = TwoSpeedSmartRouter(
    episodic_store=episodic_store,
    fact_graph=fact_graph,
    compiler=mcp_compiler,
    fallback_reasoning_fn=default_reasoning_engine,
)


@app.post("/v1/execute", response_model=AgentExecutionResponse)
def execute_agent_task(request: AgentExecutionRequest) -> AgentExecutionResponse:
    return router_engine.execute(request)


@app.post("/v1/compile", response_model=list[ShadowReplayResult])
def run_background_compiler() -> list[ShadowReplayResult]:
    all_trajectories = episodic_store.list_all()
    candidates = pattern_miner.find_candidates(all_trajectories)
    results: list[ShadowReplayResult] = []

    for candidate in candidates:
        record = mcp_compiler.compile_pattern(candidate)
        fn = mcp_compiler.get_callable(record.tool_name)
        replay_report = shadow_runner.verify_and_promote(
            tool_record=record,
            tool_callable=fn,
            historical_trajectories=candidate.trajectories,
        )
        if replay_report.promoted:
            router_engine.register_compiled_tool(record)
        results.append(replay_report)

    return results
