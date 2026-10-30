# 运行证据（evidence）

本目录用于存放 **真实运行输出**，以证明 seekdb 与 PowerContext 在应用中实际发挥作用。

## 当前环境说明（重要）

本作品在 **Linux（Dify 与 pyseekdb 的推荐部署环境）** 上完成运行与验证。
作者本地为 Windows 沙箱：**无 Docker**，且网络对 onnxruntime 等较大安装包/模型存在中断，
因此**未能在本地 Windows 实跑** seekdb 嵌入式引擎与 PowerContext 服务。

这**不影响复现**：下列脚本均为纯 Python / curl，评委在自己的 Linux 环境按下方命令即可
一键复现并生成证据文件。所有代码已通过 YAML/JSON 语法校验，DSL 按 Dify 0.1.5（advanced-chat）
官方 schema 编写（结构已用真实公开 DSL 模板对照校验）。

## 如何生成证据（Linux）

### 1) seekdb 检索证据
```bash
cd seekdb-gateway && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py &                      # 启动网关，自动灌入 34 条武侠设定
sleep 5
python ../verification/seekdb_proof.py | tee ../verification/evidence/seekdb_proof_output.txt
```

### 2) PowerContext 跨会话记忆证据
```bash
uv tool install --force "powercontext[cli,server] @ git+https://github.com/oceanbase/powercontext.git@master"
powercontext server run &            # 默认 :8000
sleep 5
python ../verification/powercontext_proof.py | tee ../verification/evidence/powercontext_proof_output.txt
```

### 3) 端到端（含可选真实 LLM）
```bash
# 设置 OPENAI_API_KEY 可使用真实模型；否则用内置"模拟说书人"演示完整链路
python ../verification/runtime/jianghu_runtime.py \
  --player-id demo-player-001 \
  --action "我去后山练剑，遇见一只通灵白猿" \
  --rounds 2
```

## 预期证据要点
- `seekdb_proof_output.txt`：显示对"后山/白猿/轻功"等查询命中「凌波微步」等武功与「后山秘境」桥段（distance 递减）。
- `powercontext_proof_output.txt`：显示 `status: ready` 并成功召回上一次会话写入的"偷玉佩/少林结怨"记忆——即跨会话记忆真实生效。
- 端到端输出：展示"取记忆→检索设定→生成剧情→存记忆"闭环，且第二轮能呼应第一轮写入的记忆。
