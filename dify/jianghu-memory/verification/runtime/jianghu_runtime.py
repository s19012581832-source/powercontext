"""
《江湖·记忆》参考运行时（不依赖 Dify UI）
--------------------------------
把 Dify Chatflow 中"取记忆 → 检索设定 → 生成剧情 → 存记忆"的完整逻辑，
用同一套代码在本地串起来，便于在无 Dify 环境也能演示、复现与收集证据。

与 Dify DSL 的节点一一对应：
  Dify 节点                        本脚本函数
  ---------------------------------- -------------------------
  创建/获取玩家记忆空间(scope)      create_scope()
  召回玩家记忆(prepare)            prepare_memory()
  seekdb-桥段素材匹配              seekdb_match_scene()
  seekdb-武学设定检索(retrieve)    seekdb_retrieve()
  续写江湖剧情(LLM)                generate_story()
  保存玩家记忆(remember)           remember()

运行（Linux，已装 pyseekdb / 已起 PowerContext，并激活 venv）：
  python verification/runtime/jianghu_runtime.py \
      --player-id demo-player-001 \
      --action "我去后山练剑，遇见一只通灵白猿" \
      --pc-url http://127.0.0.1:8000

可选：
  --rounds N            连续多轮（模拟多回合冒险）
  --openai-key sk-xxx   使用真实 OpenAI 兼容模型；缺省则使用内置"模拟说书人"
  --openai-base URL     自定义 OpenAI 兼容端点（如本地 Ollama / 通义）
  --openai-model name
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
LORE_JSON = os.path.join(HERE, "..", "..", "seekdb-gateway", "data", "lore_data.json")
SEEKDB_PATH = os.path.join(HERE, "runtime_seekdb.db")
SYS_PROMPT = (
    "你是武侠文字冒险《江湖·记忆》的江湖说书人，根据玩家行动续写有武侠味、有画面感、"
    "越聊越离奇的冒险。要求：半文半白、短句动作描写；若回忆提到玩家过往，必须自然呼应，"
    "让玩家感到你真的记得他；若提供了桥段/武学设定，请合理融入并保持世界观一致；"
    "每次续写200字以内，结尾抛出一个新的处境或选择；不替玩家做决定。"
)


# ---------------- seekdb（pyseekdb 嵌入式，等价于 Dify 的 seekdb HTTP 节点） ----------------
def get_client():
    import pyseekdb
    return pyseekdb.Client(path=SEEKDB_PATH)


def ensure_lore(client):
    with open(LORE_JSON, "r", encoding="utf-8") as f:
        lore = json.load(f)
    for name in ["sects", "skills", "scenes", "npcs"]:
        col = client.get_or_create_collection(name=name)
        items = lore[name]
        if hasattr(col, "upsert"):
            col.upsert(ids=[it["id"] for it in items],
                       documents=[it["document"] for it in items],
                       metadatas=[it.get("metadata", {}) for it in items])
        else:
            try:
                col.add(ids=[it["id"] for it in items],
                        documents=[it["document"] for it in items],
                        metadatas=[it.get("metadata", {}) for it in items])
            except Exception:
                pass
        try:
            col.refresh_index()
        except Exception as e:
            print(f"  [warn] 集合 {name} 索引刷新跳过: {e}")


def seekdb_match_scene(client, query, n=2):
    col = client.get_or_create_collection(name="scenes")
    res = col.query(query_texts=query, n_results=n)
    return _fmt(res)


def seekdb_retrieve(client, collection, query, n=3):
    col = client.get_or_create_collection(name=collection)
    res = col.query(query_texts=query, n_results=n)
    return _fmt(res)


def _fmt(res):
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    out = []
    for d, m in zip(docs, metas):
        out.append({"text": d, "meta": m})
    return out


# ---------------- PowerContext（HTTP，等价于 Dify 的 PC HTTP 节点） ----------------
def pc_post(base: str, token: str, path: str, payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{base.rstrip('/')}{path}",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"PowerContext {path} HTTP {e.code}: {detail}")


def create_scope(base, token, player_id):
    return pc_post(base, token, "/v1/scopes",
                   {"title": "江湖记忆-玩家", "summary": "玩家角色卡与剧情进度",
                    "idempotency_key": player_id})["scope_id"]


def prepare_memory(base, token, scope, query):
    return pc_post(base, token, "/v1/context/prepare",
                   {"scope_id": scope, "query": query, "max_bytes": 4000})


def remember(base, token, scope, text):
    return pc_post(base, token, "/v1/memory/remember",
                   {"scope_id": scope, "kind": "decision", "text": text,
                    "reason": "记录玩家行动以形成跨会话记忆"})


# ---------------- LLM ----------------
def openai_llm(base, key, model, sys_prompt, user_prompt):
    import urllib.request as u
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {key}"}
    payload = {"model": model, "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}]}
    req = u.Request(f"{base.rstrip('/')}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers, method="POST")
    with u.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"]


def mock_llm(action, memory, scene, skill, player):
    scene_txt = "；".join(s["meta"].get("tag", "") + ":" + s["text"][:24] for s in scene) or "（无）"
    skill_txt = "、".join(s["meta"].get("name", "") for s in skill) or "（无）"
    memory_line = ""
    if memory and memory.get("content"):
        memory_line = f"\n（你想起这人往事：{memory['content'][:60]}）"
    story = (
        f"【第N回】少侠{player}依言而行：{action}。\n"
        f"山风骤起，松涛如怒。{memory_line}\n"
        f"恰逢{sentence_pick(scene)}，周遭气氛陡然不同。你心念微动，暗运{sentence_pick2(skill_txt)}，"
        f"身形翩然，竟于间不容发之际避开一记暗袭。\n"
        f"远处传来一声长笑——这一局，才刚刚开始。\n"
        f"【说书人抛题】你欲上前查探究竟，还是暂且抽身、另寻出路？"
    )
    return story


def sentence_pick(scenes):
    if scenes:
        return scenes[0]["text"][:18] + "之景"
    return "一段奇遇"


def sentence_pick2(skill):
    return skill if skill != "（无）" else "绵劲"


# ---------------- 主流程 ----------------
def run_turn(client, base, token, scope, player, action, llm_cfg):
    print(f"\n>>> 玩家[{player}]行动：{action}")
    memory = prepare_memory(base, token, scope, action)
    scene = seekdb_match_scene(client, action, 2)
    skill = seekdb_retrieve(client, "skills", action, 3)
    print(f"    [seekdb] 命中桥段 {len(scene)} 条，武学 {len(skill)} 条")
    print(f"    [PowerContext] 召回记忆 status={memory.get('status')}")

    scene_block = json.dumps(scene, ensure_ascii=False)
    skill_block = json.dumps(skill, ensure_ascii=False)
    mem_block = memory.get("content") or "（暂无历史）"
    user_prompt = (
        f"玩家编号：{player}\n玩家本回合的行动：{action}\n\n"
        f"—— 来自过往会话的回忆 ——\n{mem_block}\n\n"
        f"—— 相关剧情桥段素材 ——\n{scene_block}\n\n"
        f"—— 相关武学/门派设定 ——\n{skill_block}\n\n"
        f"请续写这一回合的江湖剧情，并在结尾抛出一个新的处境或选择。"
    )
    if llm_cfg["key"]:
        story = openai_llm(llm_cfg["base"], llm_cfg["key"], llm_cfg["model"], SYS_PROMPT, user_prompt)
    else:
        story = mock_llm(action, memory, scene, skill, player)
    print("\n========== 说书人续写 ==========")
    print(story)
    remember(base, token, scope, f"玩家[{player}]本回合行动：{action}——说书人已据此续写剧情。")
    return story


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-id", default="demo-player-001")
    ap.add_argument("--action", default="我去后山练剑，遇见一只通灵白猿")
    ap.add_argument("--pc-url", default="http://127.0.0.1:8000")
    ap.add_argument("--pc-token", default="")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--openai-base", default="https://api.openai.com/v1")
    ap.add_argument("--openai-model", default="gpt-4o-mini")
    args = ap.parse_args()

    print(">>> 初始化 seekdb（嵌入式）")
    client = get_client()
    ensure_lore(client)

    print(">>> 连接 PowerContext")
    scope = create_scope(args.pc_url, args.pc_token, args.player_id)
    print("    scope_id =", scope)

    llm_cfg = {"key": args.openai_key, "base": args.openai_base, "model": args.openai_model}
    for i in range(args.rounds):
        act = args.action if args.rounds == 1 else f"{args.action}（第{i+1}回合）"
        run_turn(client, args.pc_url, args.pc_token, scope, args.player_id, act, llm_cfg)
    print("\n>>> 本回合已写入记忆；开新对话用同一 --player-id 即可验证跨会话记忆。")


if __name__ == "__main__":
    main()
