from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from synapseforge.compiler.pattern_miner import TrajectoryPatternCandidate
from synapseforge.models import CompiledMCPToolRecord, ScopeVisibility


class CompiledToolInputSchema(BaseModel):
    payload: dict[str, Any]


class CompiledToolOutputSchema(BaseModel):
    result: Any
    compiled_from_steps: int


class FastMCPToolCompiler:
    """Synthesizes and registers deterministic 1-step Python FastMCP tools from multi-step traces."""

    def __init__(self) -> None:
        self._callable_registry: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def compile_pattern(
        self,
        candidate: TrajectoryPatternCandidate,
        synthesized_fn: Callable[[dict[str, Any]], Any] | None = None,
        scope: ScopeVisibility = ScopeVisibility.TENANT,
    ) -> CompiledMCPToolRecord:
        tool_name = f"mcp_{candidate.task_intent.lower().replace(' ', '_')}"
        step_chain_str = " -> ".join(candidate.tool_signature)
        avg_steps = int(round(candidate.average_steps))

        source_code = (
            f"def {tool_name}(payload: dict) -> dict:\n"
            f'    """Auto-compiled FastMCP tool replacing {avg_steps}-step chain: {step_chain_str}"""\n'
            f"    return _execute_compiled_pipeline(payload)\n"
        )
        # Validate generated Python syntax tree locally
        ast.parse(source_code)

        if synthesized_fn is None:
            synthesized_fn = self._build_default_pipeline(candidate)

        def validated_wrapper(raw_payload: dict[str, Any]) -> Any:
            try:
                validated_in = CompiledToolInputSchema(payload=raw_payload)
            except ValidationError as exc:
                raise ValueError(f"Pydantic input validation failed: {exc}") from exc

            raw_out = synthesized_fn(validated_in.payload)
            validated_out = CompiledToolOutputSchema(
                result=raw_out,
                compiled_from_steps=avg_steps,
            )
            return validated_out.result

        self._callable_registry[tool_name] = validated_wrapper

        return CompiledMCPToolRecord(
            tool_name=tool_name,
            task_intent=candidate.task_intent,
            tenant_id=candidate.tenant_id,
            scope=scope,
            source_code=source_code,
            replaced_signature=candidate.tool_signature,
            shadow_pass_rate=0.0,
            is_active=False,
        )

    def get_callable(self, tool_name: str) -> Callable[[dict[str, Any]], Any]:
        if tool_name not in self._callable_registry:
            raise KeyError(f"Compiled FastMCP tool '{tool_name}' is not registered.")
        return self._callable_registry[tool_name]

    @staticmethod
    def _build_default_pipeline(
        candidate: TrajectoryPatternCandidate,
    ) -> Callable[[dict[str, Any]], Any]:
        # Maps input payload keys deterministically based on historical trajectory behavior
        lookup: dict[str, Any] = {
            repr(sorted(t.input_payload.items())): t.final_output for t in candidate.trajectories
        }

        def _pipeline(payload: dict[str, Any]) -> Any:
            key = repr(sorted(payload.items()))
            if key in lookup:
                return lookup[key]
            raise RuntimeError(f"Unrecognized payload schema in compiled tool: {payload}")

        return _pipeline
