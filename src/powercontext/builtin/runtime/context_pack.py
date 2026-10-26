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

"""Incremental context transport: full pack, prefix-reuse delta, and rebuild.

A prepared context is serialized as a :class:`ContextPack`: a deterministically
chunked list of bytes with a digest and a self-referential ``byte_count``. The
first transmission sends the full pack; later transmissions send a
:class:`ContextPackDelta` that reuses a common prefix of the previous pack and
carries only the changed suffix. The receiver rebuilds the target pack with
:func:`apply_context_pack_delta` from the wire bytes alone.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


class ContextPackError(Exception):
    """Protocol or chunking validation failure for context packs."""


@dataclass(frozen=True)
class ContextPack:
    """A complete context pack: chunked bytes plus digest and exact byte count."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items) -> ContextPack:
        items = tuple(items)
        digest = _digest_of(items)
        items_b64 = _encode_items(items)
        byte_count = _compute_byte_count(items_b64, digest)
        return cls(items=items, digest=digest, byte_count=byte_count)

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> ContextPack:
        if not isinstance(content, str):
            raise ContextPackError("content must be a string")
        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")
        data = content.encode("utf-8")
        chunks: list[bytes] = []
        index = 0
        while index < len(data):
            end = index + chunk_bytes
            if end >= len(data):
                chunks.append(data[index:])
                break
            boundary = end
            # Step back over UTF-8 continuation bytes so a chunk never splits a
            # multi-byte character in half.
            while boundary > index and (data[boundary] & 0xC0) == 0x80:
                boundary -= 1
            if boundary == index:
                raise ContextPackError(
                    "content cannot be split into chunks of at most chunk_bytes bytes"
                )
            chunks.append(data[index:boundary])
            index = boundary
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        return _encode_pack(_encode_items(self.items), self.digest, self.byte_count)

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        payload = _load_json(data)
        if not isinstance(payload, dict) or set(payload.keys()) != {
            "schema",
            "items",
            "digest",
            "byte_count",
        }:
            raise ContextPackError("pack must contain exactly schema, items, digest, and byte_count")
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("unknown pack schema")
        items = _decode_items(payload["items"], "pack items")
        digest = payload["digest"]
        if not isinstance(digest, str) or digest != _digest_of(items):
            raise ContextPackError("pack digest does not match items")
        byte_count = payload["byte_count"]
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count != len(data):
            raise ContextPackError("pack byte_count does not match payload length")
        return cls(items=items, digest=digest, byte_count=byte_count)


@dataclass(frozen=True)
class ContextPackDelta:
    """Prefix-reuse delta: keep ``prefix_items`` of base, then append ``appended``."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        limit = min(len(base.items), len(target.items))
        while prefix < limit and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        payload = {
            "schema": DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encode_items(self.appended),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        payload = _load_json(data)
        if not isinstance(payload, dict) or set(payload.keys()) != {
            "schema",
            "base_digest",
            "target_digest",
            "prefix_items",
            "appended",
        }:
            raise ContextPackError(
                "delta must contain exactly schema, base_digest, target_digest, prefix_items, and appended"
            )
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("unknown delta schema")
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if not isinstance(base_digest, str) or not isinstance(target_digest, str):
            raise ContextPackError("delta digests must be strings")
        prefix = payload["prefix_items"]
        if not isinstance(prefix, int) or isinstance(prefix, bool) or prefix < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        appended = _decode_items(payload["appended"], "delta appended")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix,
            appended=appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Validate ``delta`` against ``base`` and rebuild the target pack."""
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the base pack")
    prefix = delta.prefix_items
    if not isinstance(prefix, int) or isinstance(prefix, bool):
        raise ContextPackError("prefix_items must be a non-negative integer")
    if prefix < 0 or prefix > len(base.items):
        raise ContextPackError("prefix_items is out of range for the base pack")
    target_items = base.items[:prefix] + delta.appended
    if _digest_of(target_items) != delta.target_digest:
        raise ContextPackError("delta target digest does not match the rebuilt pack")
    return ContextPack.build(target_items)


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(raw, label: str) -> tuple[bytes, ...]:
    if not isinstance(raw, list):
        raise ContextPackError(f"{label} must be a list")
    items: list[bytes] = []
    for item in raw:
        if not isinstance(item, str):
            raise ContextPackError(f"{label} entries must be base64 strings")
        try:
            items.append(base64.b64decode(item, validate=True))
        except (binascii.Error, ValueError):
            raise ContextPackError(f"{label} entry is not valid base64") from None
    return tuple(items)


def _digest_of(items: tuple[bytes, ...]) -> str:
    """SHA-256 of the canonical ``{schema, items}`` JSON, prefixed with ``sha256:``."""
    canonical = json.dumps(
        {"items": _encode_items(items), "schema": PACK_SCHEMA},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _encode_pack(items_b64: list[str], digest: str, byte_count: int) -> bytes:
    payload = {"byte_count": byte_count, "digest": digest, "items": items_b64, "schema": PACK_SCHEMA}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _compute_byte_count(items_b64: list[str], digest: str) -> int:
    """Solve the self-referential ``byte_count == len(pack.to_bytes())`` fixed point."""
    byte_count = 0
    while True:
        encoded = _encode_pack(items_b64, digest, byte_count)
        if len(encoded) == byte_count:
            return byte_count
        byte_count = len(encoded)


def _load_json(data: bytes) -> object:
    try:
        return json.loads(data)
    except (ValueError, TypeError):
        raise ContextPackError("payload is not valid JSON") from None
