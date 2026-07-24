# Drug-Pipe

Drug-Pipe 维护一条从 MolClaw 工具图谱到 ReAct 训练与在线验证的主线：

- `tool-kg/`：taxonomy-directed 调度、Claude 单向关系裁决、canonical graph 与 grounded task。
- `data-pipe/`：任务执行、统一 evaluator、ReAct construction、LLM/hard clean 与单一 acceptance gate。
- `slime-wd/slime/drug_agent/`：共享 decision-state 派生以及 SFT、ToolRL、GAD；正式训练不执行工具。

主线数据流：

```text
MCP schema + canonical skills
  → Tool Catalog → Claude adjudication → Canonical KG
  → grounded task → real Agent execution → raw trace
  → canonical ReAct → shared decision states
  → SFT / ToolRL / GAD → checkpoint
  → explicit real-MCP evaluation/debug
```
