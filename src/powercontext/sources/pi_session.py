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

"""Pi v3 session branch capture with durable incremental checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunStatus
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"


class PiSessionInput(BaseModel):
    """One complete Pi v3 session archive and the branch leaf to resolve."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, with its full original record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


class PiEventSource(Source):
    """Captured Pi event whose identity is ``(session_id, event_id)``."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable, delimiter-safe Source identity for one Pi event."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-empty string")
    source_id = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def _record_fingerprint(record: dict[str, Any]) -> str:
    encoded = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _compact_sorted_json(arguments: Any) -> str:
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _text_of_blocks(content: Any) -> str:
    """Join the visible text of a message/custom content payload."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            if isinstance(name, str) and isinstance(call_id, str):
                parts.append(f"[tool_call {name} {call_id}] {_compact_sorted_json(arguments)}")
    return "\n".join(parts)


def _event_evidence(record: dict[str, Any]) -> tuple[str, str] | None:
    """Return ``(role, body)`` for one visible record, or ``None`` for control nodes."""
    event_type = record.get("type")
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str):
            return None
        if role == "toolResult":
            tool_name = message.get("toolName")
            tool_call_id = message.get("toolCallId")
            if not isinstance(tool_name, str) or not isinstance(tool_call_id, str):
                return None
            body = f"[tool_result {tool_name} {tool_call_id}] {_text_of_blocks(message.get('content'))}"
        elif role == "bashExecution":
            command = message.get("command")
            output = message.get("output")
            if not isinstance(command, str) or not isinstance(output, str):
                return None
            body = f"$ {command}\n{output}"
        else:
            body = _text_of_blocks(message.get("content"))
        if body == "":
            return None
        return role, body
    if event_type == "compaction":
        summary = record.get("summary")
        if not isinstance(summary, str) or summary == "":
            return None
        return "compaction", summary
    if event_type == "branch_summary":
        summary = record.get("summary")
        if not isinstance(summary, str) or summary == "":
            return None
        return "branch_summary", summary
    if event_type == "custom_message":
        body = _text_of_blocks(record.get("content"))
        if body == "":
            return None
        return "custom", body
    return None


def _parse_archive(content: bytes) -> tuple[str, list[dict[str, Any]]]:
    """Decode a Pi v3 JSONL archive into ``(session_id, records)``."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be valid UTF-8") from error
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Pi session archive must not be empty")
    parsed: list[Any] = []
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError("Pi session archive must be valid JSONL") from error
    header = parsed[0]
    if not isinstance(header, dict):
        raise ValueError("Pi session must begin with a session header record")
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("Pi session header must declare type=session and version=3")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Pi session header id must be a non-empty string")
    records: list[dict[str, Any]] = []
    for record in parsed[1:]:
        if not isinstance(record, dict):
            raise ValueError("Pi session records must be JSON objects")
        records.append(record)
    return session_id, records


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Resolve one branch to its root-to-leaf visible events, validating atomically."""
    session_id, records = _parse_archive(value.content)

    appeared: set[str] = {session_id}
    parents: dict[str, str | None] = {}
    order: list[str] = []
    for record in records:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("Pi session records must have a non-empty string id")
        if event_id in appeared:
            raise ValueError("Pi session event ids must be unique")
        parent_id = record.get("parentId")
        if parent_id is not None:
            if not isinstance(parent_id, str) or parent_id not in appeared:
                raise ValueError("Pi session parent must be empty or reference a previous node")
        appeared.add(event_id)
        parents[event_id] = parent_id
        order.append(event_id)

    leaf_id = value.leaf_id
    if leaf_id is None:
        leaf_id = order[-1]
    elif leaf_id not in parents:
        raise ValueError("Pi session leaf id must reference an existing event")

    chain: list[str] = []
    cursor: str | None = leaf_id
    while cursor is not None:
        chain.append(cursor)
        cursor = parents.get(cursor)
    chain.reverse()

    events: list[PiEventInput] = []
    for event_id in chain:
        record = next(record for record in records if record.get("id") == event_id)
        evidence = _event_evidence(record)
        if evidence is None:
            continue
        role, body = evidence
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=parents[event_id],
                event_type=record.get("type"),
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


class PiEventSourceAdapter:
    """Resolve and read the Pi event Source definition."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose each Pi event as textual evidence without coupling to ``PiEventSource``."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Incrementally capture one Pi session branch with durable deduplication."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: Path | str, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session, /) -> ConnectorRunCompletion:
        content = self.path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=content, leaf_id=self.leaf_id))

        checkpoint = session.checkpoint
        imported: dict[str, str] = {}
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("events"), dict):
            imported = dict(checkpoint["events"])
        prior_ids = set(imported)

        # Validate the full branch against prior captures before mutating anything.
        for event in events:
            fingerprint = _record_fingerprint(event.record)
            previous = imported.get(event.event_id)
            if previous is not None and previous != fingerprint:
                raise ValueError(f"Pi event {event.event_id!r} was rewritten after capture")
            imported[event.event_id] = fingerprint

        for event in events:
            if event.event_id not in prior_ids:
                await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)

        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"events": imported},
        )
