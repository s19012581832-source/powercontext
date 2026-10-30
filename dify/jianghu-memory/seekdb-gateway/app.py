"""
《江湖·记忆》 seekdb 网关服务
--------------------------------
将 pyseekdb（嵌入式，无需独立服务端）包装为 REST API，
供 Dify 的 HTTP Request 节点调用，实现对"武侠世界观设定库"的
向量 + 全文混合检索（门派 / 武功 / 桥段 / NPC）。

技术分工：
  - seekdb  -> 横向世界知识（检索 / 素材匹配），无玩家状态
  - PowerContext -> 纵向玩家记忆（角色卡 / 剧情进度），见运行说明

启动：
  pip install -r requirements.txt
  python app.py                # 默认监听 0.0.0.0:8001，自动初始化世界观数据

环境变量：
  SEEKDB_PATH   嵌入式数据库文件路径（默认 ./jianghu_seekdb.db）
  SEEKDB_PORT   服务端口（默认 8001）
  LORE_JSON     世界观数据文件路径（默认 ./data/lore_data.json）
"""
import json
import os

from flask import Flask, request, jsonify
import pyseekdb

app = Flask(__name__)

SEEKDB_PATH = os.environ.get("SEEKDB_PATH", os.path.join(os.path.dirname(__file__), "jianghu_seekdb.db"))
LORE_JSON = os.environ.get("LORE_JSON", os.path.join(os.path.dirname(__file__), "data", "lore_data.json"))
PORT = int(os.environ.get("SEEKDB_PORT", "8001"))

COLLECTIONS = ["sects", "skills", "scenes", "npcs"]

_client = None
_collections = {}


def get_client():
    global _client
    if _client is None:
        _client = pyseekdb.Client(path=SEEKDB_PATH)
    return _client


def put_docs(col, ids, documents, metadatas):
    """健壮写入：优先 upsert（可覆盖、可重复执行），否则退回 add 并忽略重复报错。"""
    if hasattr(col, "upsert"):
        col.upsert(ids=ids, documents=documents, metadatas=metadatas)
    else:
        try:
            col.add(ids=ids, documents=documents, metadatas=metadatas)
        except Exception:
            pass  # 重复 id 时忽略，保证幂等


def ensure_loaded():
    """幂等初始化：每次启动都把 lore_data.json 灌入（upsert 覆盖，可重复执行）。"""
    global _collections
    client = get_client()
    with open(LORE_JSON, "r", encoding="utf-8") as f:
        lore = json.load(f)
    for name in COLLECTIONS:
        col = client.get_or_create_collection(name=name)
        items = lore.get(name, [])
        if items:
            put_docs(
                col,
                ids=[it["id"] for it in items],
                documents=[it["document"] for it in items],
                metadatas=[it.get("metadata", {}) for it in items],
            )
            col.refresh_index()
        _collections[name] = col


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "path": SEEKDB_PATH, "collections": COLLECTIONS})


@app.route("/v1/seekdb/retrieve", methods=["POST"])
def retrieve():
    """
    检索世界观设定。
    body: {collection: str, query: str, n_results?: int, where?: dict}
    返回: {collection, results: [{id, document, metadata, distance}]}
    """
    body = request.get_json(force=True, silent=True) or {}
    collection = body.get("collection")
    query = body.get("query", "")
    n_results = int(body.get("n_results", 3))
    where = body.get("where")

    if collection not in COLLECTIONS:
        return jsonify({"error": f"unknown collection '{collection}', must be one of {COLLECTIONS}"}), 400
    if not query:
        return jsonify({"error": "query is required"}), 400

    col = _collections.get(collection) or get_client().get_or_create_collection(name=collection)
    kwargs = {"query_texts": query, "n_results": n_results}
    if where:
        kwargs["where"] = where
    try:
        res = col.query(**kwargs)
    except TypeError:
        # 部分版本不支持 where 过滤，退化为纯语义检索
        res = col.query(query_texts=query, n_results=n_results)

    results = []
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    for i in range(len(docs)):
        results.append({
            "id": ids[i] if i < len(ids) else None,
            "document": docs[i],
            "metadata": metas[i] if i < len(metas) else {},
            "distance": round(float(dists[i]), 4) if i < len(dists) else None,
        })
    return jsonify({"collection": collection, "query": query, "results": results})


@app.route("/v1/seekdb/match_scene", methods=["POST"])
def match_scene():
    """
    桥段素材匹配：根据玩家当前情境推荐合适的剧情桥段。
    body: {query: str, n_results?: int}
    """
    body = request.get_json(force=True, silent=True) or {}
    query = body.get("query", "")
    n_results = int(body.get("n_results", 2))
    if not query:
        return jsonify({"error": "query is required"}), 400
    col = _collections.get("scenes") or get_client().get_or_create_collection(name="scenes")
    res = col.query(query_texts=query, n_results=n_results)
    results = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    for i in range(len(docs)):
        results.append({
            "id": ids[i] if i < len(ids) else None,
            "document": docs[i],
            "metadata": metas[i] if i < len(metas) else {},
        })
    return jsonify({"query": query, "results": results})


if __name__ == "__main__":
    ensure_loaded()
    app.run(host="0.0.0.0", port=PORT, debug=False)
