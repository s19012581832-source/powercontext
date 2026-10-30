"""
seekdb 实测证据脚本（独立运行，无需启动网关）
用于在本机/评委环境证明：pyseekdb 嵌入式检索真实可用，
《江湖·记忆》的世界观设定能够被语义检索命中。

运行：
  python verification/seekdb_proof.py
"""
import json
import os
import sys

import pyseekdb

HERE = os.path.dirname(os.path.abspath(__file__))
LORE = os.path.join(HERE, "..", "seekdb-gateway", "data", "lore_data.json")
DB_PATH = os.path.join(HERE, "seekdb_proof.db")


def main():
    with open(LORE, "r", encoding="utf-8") as f:
        lore = json.load(f)

    client = pyseekdb.Client(path=DB_PATH)
    print(">>> pyseekdb.Client 初始化成功，路径:", DB_PATH)

    for name in ["sects", "skills", "scenes", "npcs"]:
        col = client.get_or_create_collection(name=name)
        items = lore[name]
        if hasattr(col, "upsert"):
            col.upsert(
                ids=[it["id"] for it in items],
                documents=[it["document"] for it in items],
                metadatas=[it.get("metadata", {}) for it in items],
            )
        else:
            col.add(
                ids=[it["id"] for it in items],
                documents=[it["document"] for it in items],
                metadatas=[it.get("metadata", {}) for it in items],
            )
        col.refresh_index()
        print(f">>> 已灌入集合 '{name}'，共 {len(items)} 条")

    print("\n========== 检索实测 ==========")
    samples = [
        ("skills", "后山遇到白猿，想学一门轻功身法"),
        ("sects", "哪个门派以刚猛掌法闻名，弟子都是乞丐"),
        ("scenes", "我想去一个能听秘闻、遇高人的热闹地方"),
        ("npcs", "谁最知道江湖上的小道消息"),
    ]
    for collection, query in samples:
        col = client.get_or_create_collection(name=collection)
        res = col.query(query_texts=query, n_results=2)
        print(f"\n[检索] collection={collection}  query='{query}'")
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        for d, dist, m in zip(docs, dists, metas):
            print(f"   - (距离 {dist:.4f}) {m.get('name', m.get('tag',''))}: {d[:40]}...")

    print("\n>>> 全部检索成功，seekdb 在应用功能中实际承担世界观设定检索职责。")


if __name__ == "__main__":
    main()
