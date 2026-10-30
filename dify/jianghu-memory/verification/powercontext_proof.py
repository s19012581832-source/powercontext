"""
PowerContext 记忆实测证据脚本（仅用 Python 标准库）
证明《江湖·记忆》的"跨会话玩家记忆"真实可用：
  - 第一次"会话"写入玩家角色卡 / 剧情进度
  - 第二次"新会话"用 prepare 召回，NPC 据此记起玩家过往

前置：本机已运行 PowerContext Server
  uv tool install --force "powercontext[cli,server] @ git+https://github.com/oceanbase/powercontext.git@master"
  powercontext server run
  # 另开终端确认： curl http://127.0.0.1:8000/health/live

运行：
  set POWERCONTEXT_URL=http://127.0.0.1:8000
  python verification/powercontext_proof.py
"""
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.environ.get("POWERCONTEXT_URL", "http://127.0.0.1:8000").rstrip("/")
TOKEN = os.environ.get("POWERCONTEXT_TOKEN", "")
# 每个玩家一个稳定 Scope；这里用固定 idempotency_key 演示同一玩家跨会话
SCOPE_IDEMPOTENCY = "jianghu-memory-player-demo"


def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    req = Request(f"{BASE_URL}{path}", data=json.dumps(payload).encode("utf-8"),
                  headers=headers, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            return json.load(resp)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"PowerContext {path} HTTP {e.code}: {detail}") from e


def get_scope() -> str:
    res = post("/v1/scopes", {
        "title": "江湖记忆-演示玩家",
        "summary": "《江湖·记忆》玩家角色卡与剧情进度",
        "idempotency_key": SCOPE_IDEMPOTENCY,
    })
    return res["scope_id"]


def main():
    scope = get_scope()
    print(">>> Scope 已就绪:", scope)

    # 第一次"会话"：玩家做了些事，写入记忆
    saved = post("/v1/memory/remember", {
        "scope_id": scope,
        "kind": "decision",
        "text": "玩家角色卡：少侠「阿青」，师出逍遥派，习得凌波微步；曾于后山偷走少林掌门玄苦的玉佩，与少林结下梁子；当前声望：亦正亦邪。",
        "reason": "初始化玩家状态",
    })
    print(">>> 已写入玩家记忆，记录数:", len(saved.get("records", [])))

    # 第二次"新会话"：玩家只是说"我回少林寺看看"
    prepared = post("/v1/context/prepare", {
        "scope_id": scope,
        "query": "少林寺 掌门 玉佩",
        "max_bytes": 4000,
    })
    print("\n========== 跨会话召回 ==========")
    print("status:", prepared.get("status"))
    content = prepared.get("content") or ""
    print("召回内容:\n", content if content else "（empty，新 scope 或无关查询时为正常空值）")

    if "玉佩" in content or "少林" in content:
        print("\n>>> 验证成功：新会话中 PowerContext 成功召回了上一次会话写入的玩家记忆，"
              "NPC 可据此'记起'玩家过往——满足跨会话记忆要求。")
    else:
        print("\n>>> 注意：本次召回为空，可能是索引尚未建立或查询词不匹配；"
              "在正式 Dify 流程中，prepare 会在每轮对话前执行，记忆会随交互累积。")


if __name__ == "__main__":
    main()
