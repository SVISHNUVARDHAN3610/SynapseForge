from __future__ import annotations

import json
from typing import Any
from uuid import uuid4


class SmartContextManager:
    """Offloads oversized raw data payloads from the active prompt into local key-value state."""

    def __init__(self, max_inline_chars: int = 256) -> None:
        self.max_inline_chars = max_inline_chars
        self._blob_store: dict[str, Any] = {}

    def ingest_payload(self, payload: Any) -> dict[str, Any]:
        serialized = json.dumps(payload, default=str)
        if len(serialized) <= self.max_inline_chars:
            return {
                "offloaded": False,
                "inline_data": payload,
                "char_length": len(serialized),
            }

        ref_key = f"ctx_blob_{uuid4().hex[:10]}"
        self._blob_store[ref_key] = payload
        schema_summary = self._summarize_structure(payload)
        return {
            "offloaded": True,
            "ref_key": ref_key,
            "schema_summary": schema_summary,
            "preview": serialized[:120] + "...",
            "original_char_length": len(serialized),
        }

    def resolve_ref(self, ref_key: str) -> Any:
        if ref_key not in self._blob_store:
            raise KeyError(f"Context reference '{ref_key}' not found in store.")
        return self._blob_store[ref_key]

    @staticmethod
    def _summarize_structure(payload: Any) -> str:
        if isinstance(payload, dict):
            keys = list(payload.keys())[:8]
            return f"dict(keys={keys}, total_keys={len(payload)})"
        if isinstance(payload, list):
            elem_type = type(payload[0]).__name__ if payload else "empty"
            return f"list(len={len(payload)}, item_type={elem_type})"
        return type(payload).__name__
