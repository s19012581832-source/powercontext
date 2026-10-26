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

"""HTTP scoring-service Memory reranker with cross-batch global ordering."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score coarse candidates through an offline HTTP service and reorder globally."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        if not candidates:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > len(candidates):
            raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003

        scores: dict[int, float] = {}
        request_count = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            payload = {
                "query": query,
                "candidates": [
                    {"rank": start + index + 1, "text": candidate.text}
                    for index, candidate in enumerate(batch)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            request_count += 1
            expected = {start + index + 1 for index in range(len(batch))}
            scores.update(_parse_scores(response, expected))

        ordered = sorted(scores, key=lambda rank: (-scores[rank], rank))
        return MemoryRerankDecision(
            selected_ranks=tuple(ordered[:limit]),
            usage=InferenceUsage(requests=request_count),
        )


def _parse_scores(response: httpx.Response, expected: set[int]) -> dict[int, float]:
    try:
        data = response.json()
    except ValueError as error:
        raise ValueError("rerank response is not valid JSON") from error  # noqa: TRY003
    if not isinstance(data, dict) or set(data.keys()) != {"scores"}:
        raise ValueError("rerank response must contain only a scores field")  # noqa: TRY003
    scores = data["scores"]
    if not isinstance(scores, list):
        raise ValueError("rerank response scores must be a list")  # noqa: TRY003

    seen: set[int] = set()
    result: dict[int, float] = {}
    for item in scores:
        if not isinstance(item, dict) or set(item.keys()) != {"rank", "score"}:
            raise ValueError("each score record must contain only rank and score")  # noqa: TRY003
        rank = item["rank"]
        score = item["score"]
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("score rank must be an integer")  # noqa: TRY003
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("score must be a number")  # noqa: TRY003
        if not math.isfinite(score):
            raise ValueError("score must be finite")  # noqa: TRY003
        if rank not in expected or rank in seen:
            raise ValueError("response ranks must cover the batch exactly once")  # noqa: TRY003
        seen.add(rank)
        result[rank] = float(score)

    if seen != expected:
        raise ValueError("response ranks must cover the batch exactly once")  # noqa: TRY003
    return result
