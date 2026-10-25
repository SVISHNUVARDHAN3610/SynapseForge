from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from synapseforge.compiler.mcp_compiler import FastMCPToolCompiler
from synapseforge.compiler.pattern_miner import TrajectoryPatternMiner
from synapseforge.compiler.shadow_replay import LocalShadowReplayRunner
from synapseforge.memory.context_manager import SmartContextManager
from synapseforge.memory.episodic import EpisodicTrajectoryStore
from synapseforge.memory.temporal_graph import BiTemporalFactGraph
from synapseforge.models import (
    AgentExecutionRequest,
    EpisodicTrajectory,
    RoutingTier,
    ScopeVisibility,
    ToolStepSpan,
)
from synapseforge.orchestration.router import TwoSpeedSmartRouter
from synapseforge.orchestration.telemetry import compute_telemetry_summary


def _make_eight_step_trajectory(
    tenant_id: str,
    user_id: int,
    base_amount: int,
) -> EpisodicTrajectory:
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
    steps = [
        ToolStepSpan(
            step_index=idx,
            tool_name=name,
            tool_input={"user_id": user_id, "base_amount": base_amount},
            tool_output={"ok": True},
            latency_ms=40.0,
            prompt_tokens=220,
        )
        for idx, name in enumerate(step_names)
    ]
    return EpisodicTrajectory(
        tenant_id=tenant_id,
        task_intent="reconcile_user_metrics",
        user_query=f"Reconcile annual billing metrics for user {user_id}",
        input_payload={"user_id": user_id, "base_amount": base_amount},
        steps=steps,
        final_output={
            "user_id": user_id,
            "reconciled_mrr": base_amount * 12,
            "status": "VERIFIED",
        },
        succeeded=True,
    )


def test_smart_context_manager_offloads_large_payloads() -> None:
    manager = SmartContextManager(max_inline_chars=64)

    small_summary = manager.ingest_payload({"ok": True})
    assert small_summary["offloaded"] is False
    assert small_summary["inline_data"] == {"ok": True}

    large_payload = {"records": [{"id": i, "value": "x" * 30} for i in range(20)]}
    large_summary = manager.ingest_payload(large_payload)
    assert large_summary["offloaded"] is True
    assert "ref_key" in large_summary
    assert "dict(keys=['records']" in large_summary["schema_summary"]

    restored = manager.resolve_ref(large_summary["ref_key"])
    assert restored == large_payload


def test_bitemporal_fact_graph_conflict_resolution_and_rbac() -> None:
    graph = BiTemporalFactGraph()
    t0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    t1 = t0 + timedelta(days=10)

    # Initial fact at t0
    fact_v1 = graph.upsert_fact(
        tenant_id="tenant_alpha",
        subject="billing_api",
        predicate="rate_limit_per_min",
        object_value="100",
        scope=ScopeVisibility.TENANT,
        timestamp=t0,
    )
    assert fact_v1.is_active is True

    # Updated fact at t1 expires fact_v1
    fact_v2 = graph.upsert_fact(
        tenant_id="tenant_alpha",
        subject="billing_api",
        predicate="rate_limit_per_min",
        object_value="500",
        scope=ScopeVisibility.TENANT,
        timestamp=t1,
    )
    assert fact_v1.valid_to == t1
    assert fact_v2.is_active is True

    # Point-in-time query between t0 and t1 returns v1
    mid_point = t0 + timedelta(days=5)
    historical = graph.query_active_facts("tenant_alpha", subject="billing_api", as_of=mid_point)
    assert len(historical) == 1
    assert historical[0].object_value == "100"

    # Current query returns only v2
    current = graph.query_active_facts("tenant_alpha", subject="billing_api")
    assert len(current) == 1
    assert current[0].object_value == "500"

    # Tenant isolation check: tenant_beta cannot see tenant_alpha's private facts
    beta_facts = graph.query_active_facts("tenant_beta", subject="billing_api")
    assert len(beta_facts) == 0


def test_pattern_mining_compilation_and_25_task_shadow_replay() -> None:
    store = EpisodicTrajectoryStore()
    for i in range(1, 26):
        store.log_trajectory(_make_eight_step_trajectory("tenant_alpha", user_id=i, base_amount=i * 10))

    miner = TrajectoryPatternMiner(min_steps=4, min_occurrences=5)
    candidates = miner.find_candidates(store.list_all("tenant_alpha"))
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.average_steps == 8.0

    compiler = FastMCPToolCompiler()
    record = compiler.compile_pattern(
        candidate,
        synthesized_fn=lambda payload: {
            "user_id": payload["user_id"],
            "reconciled_mrr": payload["base_amount"] * 12,
            "status": "VERIFIED",
        },
    )

    runner = LocalShadowReplayRunner(replay_sample_size=25, required_pass_rate=1.0)
    report = runner.verify_and_promote(
        tool_record=record,
        tool_callable=compiler.get_callable(record.tool_name),
        historical_trajectories=candidate.trajectories,
    )

    assert report.tasks_evaluated == 25
    assert report.tasks_passed == 25
    assert report.pass_rate == 1.0
    assert report.promoted is True
    assert record.is_active is True
    # 8 steps * 220 tokens = 1760 tokens -> 120 tokens (>93% reduction, exceeding 65% target)
    assert report.token_reduction_pct > 65.0


def test_shadow_replay_rejects_divergent_compiled_tool() -> None:
    trajectories = [
        _make_eight_step_trajectory("tenant_alpha", user_id=i, base_amount=100)
        for i in range(1, 26)
    ]
    miner = TrajectoryPatternMiner(min_steps=4, min_occurrences=5)
    candidate = miner.find_candidates(trajectories)[0]

    compiler = FastMCPToolCompiler()
    # Buggy multiplier (10 instead of 12)
    record = compiler.compile_pattern(
        candidate,
        synthesized_fn=lambda payload: {
            "user_id": payload["user_id"],
            "reconciled_mrr": payload["base_amount"] * 10,
            "status": "VERIFIED",
        },
    )

    runner = LocalShadowReplayRunner(replay_sample_size=25, required_pass_rate=1.0)
    report = runner.verify_and_promote(
        tool_record=record,
        tool_callable=compiler.get_callable(record.tool_name),
        historical_trajectories=candidate.trajectories,
    )

    assert report.tasks_passed == 0
    assert report.promoted is False
    assert record.is_active is False


def test_two_speed_router_and_automatic_self_healing_fallback() -> None:
    store = EpisodicTrajectoryStore()
    graph = BiTemporalFactGraph()
    compiler = FastMCPToolCompiler()

    graph.upsert_fact(
        tenant_id="tenant_alpha",
        subject="reconciliation_policy",
        predicate="currency",
        object_value="USD",
    )

    for i in range(1, 26):
        store.log_trajectory(_make_eight_step_trajectory("tenant_alpha", user_id=i, base_amount=50))

    candidate = TrajectoryPatternMiner().find_candidates(store.list_all("tenant_alpha"))[0]
    api_version = {"v": 1}

    def dynamic_tool(payload: dict[str, Any]) -> dict[str, Any]:
        if api_version["v"] != 1:
            raise RuntimeError("Upstream API schema changed: missing v2 field")
        return {
            "user_id": payload["user_id"],
            "reconciled_mrr": payload["base_amount"] * 12,
            "status": "VERIFIED",
        }

    record = compiler.compile_pattern(candidate, synthesized_fn=dynamic_tool)
    LocalShadowReplayRunner().verify_and_promote(
        record,
        compiler.get_callable(record.tool_name),
        candidate.trajectories,
    )

    def fallback_reasoning(
        req: AgentExecutionRequest,
        facts: list[str],
    ) -> tuple[dict[str, Any], list[ToolStepSpan]]:
        traj = _make_eight_step_trajectory(
            req.tenant_id,
            req.input_payload["user_id"],
            req.input_payload["base_amount"],
        )
        return traj.final_output, traj.steps

    router = TwoSpeedSmartRouter(
        episodic_store=store,
        fact_graph=graph,
        compiler=compiler,
        fallback_reasoning_fn=fallback_reasoning,
        min_tool_health_rate=0.80,
    )
    router.register_compiled_tool(record)

    # 1. Healthy compiled tool runs on Tier 1 Fast Path in 1 step
    req1 = AgentExecutionRequest(
        tenant_id="tenant_alpha",
        task_intent="reconcile_user_metrics",
        user_query="Reconcile metrics for user 99",
        input_payload={"user_id": 99, "base_amount": 50},
    )
    res1 = router.execute(req1)
    assert res1.routing_tier == RoutingTier.TIER_1_FAST_MCP
    assert res1.steps_taken == 1
    assert res1.prompt_tokens_used == 120

    # 2. Simulate upstream API change breaking the compiled tool
    api_version["v"] = 2
    res2 = router.execute(req1)
    assert res2.routing_tier == RoutingTier.TIER_2_REASONING
    assert res2.steps_taken == 8
    assert res2.self_healing_triggered is True
    assert record.is_active is False
    assert record.tool_name in router.recompilation_queue
    assert "reconciliation_policy.currency=USD" in res2.active_facts_injected

    # 3. Telemetry summary verification
    baseline = store.list_all("tenant_alpha")[:25]
    summary = compute_telemetry_summary(baseline, [store.list_all("tenant_alpha")[25]])
    assert summary.baseline_avg_steps == 8.0
    assert summary.current_avg_steps == 1.0
    assert summary.token_savings_pct > 65.0
