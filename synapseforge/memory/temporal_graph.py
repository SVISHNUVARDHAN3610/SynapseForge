from __future__ import annotations

from datetime import UTC, datetime

from synapseforge.models import ScopeVisibility, TemporalFact


class BiTemporalFactGraph:
    """Time-aware knowledge graph supporting valid_from / valid_to conflict resolution and RBAC."""

    def __init__(self) -> None:
        self._facts: list[TemporalFact] = []

    def upsert_fact(
        self,
        tenant_id: str,
        subject: str,
        predicate: str,
        object_value: str,
        scope: ScopeVisibility = ScopeVisibility.TENANT,
        timestamp: datetime | None = None,
    ) -> TemporalFact:
        now = timestamp or datetime.now(UTC)

        for existing in self._facts:
            same_scope_key = (
                existing.subject == subject
                and existing.predicate == predicate
                and existing.is_active
                and (
                    (scope == ScopeVisibility.GLOBAL and existing.scope == ScopeVisibility.GLOBAL)
                    or (existing.tenant_id == tenant_id and existing.scope == ScopeVisibility.TENANT)
                )
            )
            if same_scope_key:
                if existing.object_value == object_value:
                    return existing
                existing.valid_to = now

        new_fact = TemporalFact(
            tenant_id=tenant_id,
            scope=scope,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            valid_from=now,
            valid_to=None,
        )
        self._facts.append(new_fact)
        return new_fact

    def query_active_facts(
        self,
        tenant_id: str,
        subject: str | None = None,
        as_of: datetime | None = None,
    ) -> list[TemporalFact]:
        results: list[TemporalFact] = []
        for fact in self._facts:
            accessible = (
                fact.scope == ScopeVisibility.GLOBAL or fact.tenant_id == tenant_id
            )
            if not accessible:
                continue
            if subject is not None and fact.subject != subject:
                continue

            if as_of is None:
                if fact.is_active:
                    results.append(fact)
            else:
                started = fact.valid_from <= as_of
                not_expired = fact.valid_to is None or fact.valid_to > as_of
                if started and not_expired:
                    results.append(fact)

        return results

    def get_history(self, tenant_id: str, subject: str, predicate: str) -> list[TemporalFact]:
        return [
            f
            for f in self._facts
            if (f.tenant_id == tenant_id or f.scope == ScopeVisibility.GLOBAL)
            and f.subject == subject
            and f.predicate == predicate
        ]
