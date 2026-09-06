# Drug-Pipe

Drug-Pipe 维护 MolClaw 任务构造、Claude 数据采集和 Qwen3.5 structured SFT 主线：

- `tool-kg/`：taxonomy-directed 调度、Claude 单向关系裁决、canonical graph 与 grounded task。
- `data-pipe/`：任务执行、immutable raw、semantic canonical、Python/LLM clean 与 Qwen3.5 SFT materialization。
- `slime-wd/slime/drug_agent/`：structured-message SFT 及尚待专项整改的 ToolRL/GAD；正式训练不执行工具。

主线数据流：

```text
MCP schema + canonical skills
  → Tool Catalog → Claude adjudication → Canonical KG
  → grounded task → real Agent execution → immutable raw
  → semantic canonical → Python clean → mandatory LLM reasoning clean
  → Qwen3.5 structured messages + tools → Slime SFT → checkpoint
  → explicit real-MCP evaluation/debug
```

Drug-Pipe 不再为 SFT 生成或解析 `<thought>/<tool_call>/<observation>/<final_answer>`。旧 XML
ToolRL/online 代码属于后续 ToolRL 专项范围，不是当前 SFT 文档或启动入口。
