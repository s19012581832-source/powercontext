# 《江湖·记忆》— 会记住你的武侠文字冒险

OceanBase 大赛赛题 **Just for Fun——用 Dify 做一个好玩的智能应用** 的参赛作品。

- **Dify（必选）**：核心流程编排（Chatflow DSL）
- **seekdb**：武侠世界观设定检索（门派 / 武功 / 桥段 / NPC），向量+语义混合检索
- **PowerContext**：跨会话玩家记忆，让 NPC 在新对话里依然记得你

## 目录结构

```
dify/jianghu-memory/
├── README.md                  # 本文件
├── dsl/
│   └── jianghu_memory.yaml    # Dify DSL（advanced-chat），提交主产物
├── seekdb-gateway/            # seekdb 接入层（Flask 包装 pyseekdb 嵌入式）
│   ├── app.py                 # REST 网关：/v1/seekdb/retrieve、/v1/seekdb/match_scene、/health
│   ├── requirements.txt
│   └── data/lore_data.json    # 武侠世界观设定库（34 条）
├── docs/
│   ├── 运行说明.md            # 环境、部署、配置、分工
│   └── 验证样例.md            # 可逐步执行的验证样例与证据清单
└── verification/
    ├── seekdb_proof.py        # seekdb 检索独立实测脚本
    ├── powercontext_proof.py  # PowerContext 记忆独立实测脚本
    ├── runtime/
    │   └── jianghu_runtime.py # 不依赖 Dify UI 的参考运行时（同套逻辑端到端）
    └── evidence/              # 真实运行输出（Linux 上由上述脚本生成）
```

## 快速体验（Linux）

```bash
# 1) seekdb 网关
cd seekdb-gateway && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python app.py   # :8001

# 2) PowerContext
uv tool install --force "powercontext[cli,server] @ git+https://github.com/oceanbase/powercontext.git@master"
powercontext server run                       # :8000

# 3) Dify：导入 dsl/jianghu_memory.yaml，配置环境变量 PC_URL/SEEKDB_URL/PC_TOKEN 与模型，开始对话
```

详见 `docs/运行说明.md` 与 `docs/验证样例.md`。
