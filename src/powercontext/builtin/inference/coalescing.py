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

"""Coalesce, batch, and cache calls to an :class:`EmbeddingModel`.

Concurrent callers share one provider request per distinct text: repeated texts
are embedded once, distinct texts are grouped into batches bounded by
``max_batch_size``, and recently embedded texts are served from a bounded LRU
cache. Each caller receives its result in the original order, and its usage is
attributed to the provider batches that actually contributed to that result.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


class CoalescingEmbeddingModel:
    """Deduplicate, batch, and cache an :class:`EmbeddingModel` provider."""

    def __init__(
        self,
        base: EmbeddingModel,
        *,
        max_batch_size: int = 32,
        coalesce_ms: float = 0.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must be non-negative")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        self._base = base
        self.profile = base.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[EmbeddingVector]] = {}
        self._pending: list[str] = []
        self._flushing = False
        self._next_batch_id = 0
        self._batch_usage: dict[int, InferenceUsage] = {}
        self._text_batch: dict[str, int] = {}

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        vectors: list[EmbeddingVector | None] = [None] * len(texts)
        awaiters: dict[asyncio.Future[EmbeddingVector], list[int]] = {}
        missed: set[str] = set()
        for index, text in enumerate(texts):
            vector = self._cache_get(text)
            if vector is not None:
                vectors[index] = vector
                continue
            future = self._inflight.get(text)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                self._inflight[text] = future
                self._pending.append(text)
                self._schedule_flush()
            missed.add(text)
            awaiters.setdefault(future, []).append(index)
        for future in awaiters:
            # Shield so cancelling one waiter never cancels the shared provider
            # future that other callers are still awaiting.
            await asyncio.shield(future)
        for future, indices in awaiters.items():
            vector = future.result()
            for index in indices:
                vectors[index] = vector
        return EmbeddingResult(vectors=tuple(vectors), usage=self._usage_for(missed))

    def _usage_for(self, texts: set[str]) -> InferenceUsage:
        batch_ids: set[int] = set()
        for text in texts:
            batch_id = self._text_batch.get(text)
            if batch_id is not None:
                batch_ids.add(batch_id)
        if not batch_ids:
            return InferenceUsage(requests=0)
        requests = 0
        input_tokens: int | None = None
        output_tokens: int | None = None
        for batch_id in batch_ids:
            usage = self._batch_usage[batch_id]
            requests += usage.requests
            if usage.input_tokens is not None:
                input_tokens = (input_tokens or 0) + usage.input_tokens
            if usage.output_tokens is not None:
                output_tokens = (output_tokens or 0) + usage.output_tokens
        return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)

    def _cache_get(self, text: str) -> EmbeddingVector | None:
        if self._cache_size <= 0:
            return None
        vector = self._cache.get(text)
        if vector is None:
            return None
        self._cache.move_to_end(text)
        return vector

    def _cache_put(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size <= 0:
            return
        self._cache.pop(text, None)
        self._cache[text] = vector
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _schedule_flush(self) -> None:
        if self._flushing:
            return
        self._flushing = True
        asyncio.get_running_loop().create_task(self._flush())

    async def _flush(self) -> None:
        try:
            if self._coalesce_seconds > 0:
                await asyncio.sleep(self._coalesce_seconds)
            while self._pending:
                batch: list[str] = []
                while self._pending and len(batch) < self._max_batch_size:
                    batch.append(self._pending.pop(0))
                await self._dispatch(batch)
        finally:
            self._flushing = False
            if self._pending:
                self._schedule_flush()

    async def _dispatch(self, batch: list[str]) -> None:
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        try:
            result = await self._base.embed(tuple(batch))
        except BaseException as error:
            for text in batch:
                future = self._inflight.pop(text, None)
                if future is not None and not future.done():
                    future.set_exception(error)
            return
        vectors = result.vectors
        if len(vectors) != len(batch):
            error = ValueError("embedding provider returned an unexpected number of vectors")
            for text in batch:
                future = self._inflight.pop(text, None)
                if future is not None and not future.done():
                    future.set_exception(error)
            return
        self._batch_usage[batch_id] = result.usage
        for text, vector in zip(batch, vectors):
            self._text_batch[text] = batch_id
            self._cache_put(text, vector)
            future = self._inflight.pop(text, None)
            if future is not None and not future.done():
                future.set_result(vector)
