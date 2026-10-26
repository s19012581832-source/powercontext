# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Source-driven lifecycle of derived Memory and Topic Memory evidence.

A scope registers a directed dependency graph whose leaves are captured Sources
and whose interior nodes are Memory entries and Topic Memory artifacts. Each
interior node names its evidence parents and a policy (``all`` or ``any``). A
Source is authoritative: its ``available`` flag is driven only by
:class:`SourceLifecycleEvent` values, and the reachable active set is recomputed
after every accepted event. Memory entries are deactivated or reactivated through
the Memory service's own state mechanism; Topic Memory hits are filtered at read
time, since Topic artifacts carry no inactive state of their own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef
from powercontext.artifacts.models import MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.sources import SourceRef

_Reference = SourceRef | MemoryCitation | ArtifactRef
_Policy = Literal["all", "any"]


class SourceLifecycleError(Exception):
    """A Source lifecycle event, registration, or resolution was rejected."""


class SourceLifecycleEvent(BaseModel):
    """One authoritative availability transition for a registered Source node."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool
    reason: str

    @field_validator("node_id")
    @classmethod
    def _validate_node_id(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("node_id must be a non-empty trimmed string")
        return value


@dataclass
class _LifecycleNode:
    node_id: str
    family: str
    reference: _Reference
    parents: tuple[str, ...]
    policy: str
    revision: int | None = None
    available: bool | None = None
    last_sequence: int | None = None


def _dump_reference(reference: _Reference) -> bytes:
    return reference.model_dump_json().encode()


def _load_reference(family: str, payload: bytes) -> _Reference:
    data = json.loads(payload.decode("utf-8"))
    if family == "source":
        return SourceRef.model_validate(data)
    if family == "memory":
        return MemoryCitation.model_validate(data)
    return ArtifactRef.model_validate(data)


def _dump_parents(parents: tuple[str, ...]) -> bytes:
    return json.dumps(list(parents), separators=(",", ":")).encode()


class ScopedSourceLifecycle:
    """Register and drive one scope's Source-to-evidence dependency graph."""

    def __init__(self, runtime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id

    @property
    def _database(self) -> AsyncDatabase:
        database = self._runtime._database
        if database is None:
            raise SourceLifecycleError("lifecycle persistence is unavailable")
        return database

    async def register_source(self, node_id: str, source_ref: SourceRef, /, *, revision: int) -> None:
        if type(revision) is not int or revision < 1:
            raise SourceLifecycleError("revision must be a positive integer")
        async with self._runtime._scope_operation(self.scope_id):
            async with self._runtime._locked(self.scope_id):
                await self._register(
                    node_id,
                    family="source",
                    reference=source_ref,
                    parents=(),
                    policy="source",
                    revision=revision,
                    available=True,
                    last_sequence=None,
                )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: tuple[str, ...],
        policy: _Policy,
    ) -> None:
        self._validate_policy(policy)
        async with self._runtime._scope_operation(self.scope_id):
            async with self._runtime._locked(self.scope_id):
                await self._register(
                    node_id,
                    family="memory",
                    reference=citation,
                    parents=parents,
                    policy=policy,
                    revision=None,
                    available=None,
                    last_sequence=None,
                )
                async with self._database.transaction() as connection:
                    nodes = await self._load(connection)
                if node_id not in self._active_nodes(nodes):
                    await self._apply_memory_transition(nodes, deactivate_ids=[node_id], reactivate_ids=[])

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        /,
        *,
        parents: tuple[str, ...],
        policy: _Policy,
    ) -> None:
        self._validate_policy(policy)
        async with self._runtime._scope_operation(self.scope_id):
            async with self._runtime._locked(self.scope_id):
                await self._register(
                    node_id,
                    family="topic",
                    reference=reference,
                    parents=parents,
                    policy=policy,
                    revision=None,
                    available=None,
                    last_sequence=None,
                )

    @staticmethod
    def _validate_policy(policy: str) -> None:
        if policy not in {"all", "any"}:
            raise SourceLifecycleError("policy must be 'all' or 'any'")

    async def _register(
        self,
        node_id: str,
        *,
        family: str,
        reference: _Reference,
        parents: tuple[str, ...],
        policy: str,
        revision: int | None,
        available: bool | None,
        last_sequence: int | None,
    ) -> None:
        if not node_id or node_id != node_id.strip():
            raise SourceLifecycleError("node_id must be a non-empty trimmed string")
        if family == "source":
            normalized_parents: tuple[str, ...] = ()
        else:
            normalized_parents = tuple(parents)
            if not normalized_parents and policy == "any":
                raise SourceLifecycleError("a node with 'any' policy requires at least one parent")
        async with self._database.transaction() as connection:
            existing = await self._load(connection)
            if node_id in existing:
                raise SourceLifecycleError(f"node already registered: {node_id}")
            if self._creates_cycle(node_id, normalized_parents, existing):
                raise SourceLifecycleError(f"registration would create a dependency cycle: {node_id}")
            missing = [parent for parent in normalized_parents if parent not in existing]
            if missing:
                raise SourceLifecycleError(f"unknown parent nodes: {missing}")
            await connection.execute(
                insert(SOURCE_LIFECYCLE_NODES_TABLE).values(
                    scope_id=self.scope_id,
                    node_id=node_id,
                    family=family,
                    reference=_dump_reference(reference),
                    parents=_dump_parents(normalized_parents),
                    policy=policy,
                    revision=revision,
                    available=available,
                    last_sequence=last_sequence,
                )
            )

    @staticmethod
    def _creates_cycle(node_id: str, parents: tuple[str, ...], nodes: dict[str, _LifecycleNode]) -> bool:
        # The new node does not exist yet, so the only reachable cycle is one that
        # returns to ``node_id`` through already-registered ancestors.
        for parent in parents:
            if parent == node_id:
                return True
            stack = [parent]
            seen: set[str] = set()
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                if current == node_id:
                    return True
                ancestor = nodes.get(current)
                if ancestor is not None:
                    stack.extend(ancestor.parents)
        return False

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        async with self._runtime._scope_operation(self.scope_id):
            async with self._runtime._locked(self.scope_id):
                async with self._database.transaction() as connection:
                    nodes = await self._load(connection)
                    node = nodes.get(event.node_id)
                    if node is None or node.family != "source":
                        raise SourceLifecycleError(f"unknown source node: {event.node_id}")
                    if event.revision != node.revision:
                        raise SourceLifecycleError("event revision does not match the registered source revision")
                    last_sequence = 0 if node.last_sequence is None else node.last_sequence
                    if event.sequence < last_sequence:
                        raise SourceLifecycleError("event sequence is older than the last applied sequence")
                    if event.sequence == last_sequence:
                        if event.available == node.available:
                            return sorted(self._active_nodes(nodes))
                        raise SourceLifecycleError("event sequence is already applied with a different availability")
                    old_active = self._active_nodes(nodes)
                    nodes[event.node_id] = _LifecycleNode(
                        node_id=node.node_id,
                        family=node.family,
                        reference=node.reference,
                        parents=node.parents,
                        policy=node.policy,
                        revision=node.revision,
                        available=event.available,
                        last_sequence=event.sequence,
                    )
                    await connection.execute(
                        update(SOURCE_LIFECYCLE_NODES_TABLE)
                        .where(
                            SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self.scope_id,
                            SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == event.node_id,
                        )
                        .values(available=event.available, last_sequence=event.sequence)
                    )
                    new_active = self._active_nodes(nodes)
                await self._toggle_memory(nodes, old_active, new_active)
                return sorted(new_active)

    async def resolve(self, node_id: str, /, *, purpose: str) -> _Reference:
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("purpose must be 'generation' or 'audit'")
        async with self._database.transaction() as connection:
            nodes = await self._load(connection)
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError(f"unknown node: {node_id}")
        if purpose == "generation" and node_id not in self._active_nodes(nodes):
            raise SourceLifecycleError(f"node is not available for generation: {node_id}")
        return node.reference

    async def inactive_topic_refs(self) -> list[ArtifactRef]:
        """Return Topic references whose nodes are currently inactive."""
        async with self._database.transaction() as connection:
            nodes = await self._load(connection)
        active = self._active_nodes(nodes)
        return [
            node.reference
            for node_id, node in nodes.items()
            if node.family == "topic" and node_id not in active and isinstance(node.reference, ArtifactRef)
        ]

    @staticmethod
    def _active_nodes(nodes: dict[str, _LifecycleNode]) -> set[str]:
        memo: dict[str, bool] = {}

        def is_active(node_id: str) -> bool:
            if node_id in memo:
                return memo[node_id]
            node = nodes[node_id]
            if node.family == "source":
                result = bool(node.available)
            else:
                support = [is_active(parent) for parent in node.parents]
                result = all(support) if node.policy == "all" else any(support)
            memo[node_id] = result
            return result

        return {node_id for node_id in nodes if is_active(node_id)}

    async def _toggle_memory(
        self,
        nodes: dict[str, _LifecycleNode],
        old_active: set[str],
        new_active: set[str],
    ) -> None:
        deactivate = sorted(
            node_id
            for node_id in old_active - new_active
            if nodes[node_id].family == "memory"
        )
        reactivate = sorted(
            node_id
            for node_id in new_active - old_active
            if nodes[node_id].family == "memory"
        )
        await self._apply_memory_transition(nodes, deactivate_ids=deactivate, reactivate_ids=reactivate)

    async def _apply_memory_transition(
        self,
        nodes: dict[str, _LifecycleNode],
        *,
        deactivate_ids: list[str],
        reactivate_ids: list[str],
    ) -> None:
        if not deactivate_ids and not reactivate_ids:
            return
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            current = await service.head(context.artifacts.memory_artifact_id)
            if deactivate_ids:
                versions = [await service.validate_citation(nodes[node_id].reference) for node_id in deactivate_ids]
                current = await service.forget(current, entries=versions, reason="source lifecycle withdrawal")
            if reactivate_ids:
                versions = [await service.validate_citation(nodes[node_id].reference) for node_id in reactivate_ids]
                current = await service.reactivate(current, entries=versions, reason="source lifecycle restore")

    async def _load(self, connection: AsyncConnection) -> dict[str, _LifecycleNode]:
        rows = (
            await connection.execute(
                select(SOURCE_LIFECYCLE_NODES_TABLE).where(
                    SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self.scope_id,
                )
            )
        ).mappings().all()
        nodes: dict[str, _LifecycleNode] = {}
        for row in rows:
            node_id = str(row["node_id"])
            family = str(row["family"])
            nodes[node_id] = _LifecycleNode(
                node_id=node_id,
                family=family,
                reference=_load_reference(family, bytes(row["reference"])),
                parents=tuple(json.loads(bytes(row["parents"]).decode("utf-8"))),
                policy=str(row["policy"]),
                revision=None if row["revision"] is None else int(row["revision"]),
                available=None if row["available"] is None else bool(row["available"]),
                last_sequence=None if row["last_sequence"] is None else int(row["last_sequence"]),
            )
        return nodes


class SourceLifecycleApplication:
    """Select the scoped Source Lifecycle application service."""

    def __init__(self, runtime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, scope_id)
