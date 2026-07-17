# Drug-Pipe

Drug-Pipe 维护一条从 MolClaw 工具图谱到 ReAct 训练与在线验证的主线：

- `tool-kg/`：工具目录、Claude 单向关系裁决、canonical graph 与 grounded task。
- `data-pipe/`：任务执行、统一 benchmark evaluator、raw trace 与 deterministic ReAct curation。
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

永久事实文档只有：

- [主线职责与 authority](docs/MAINLINE.md)
- [canonical 数据格式](docs/DATA_FORMATS.md)
- [当前运行命令](docs/RUN_COMMANDS.md)
- [主线最小运行检查](docs/RUN_MINITEST.md)
- [仍存在的问题](docs/KNOWN_ISSUES.md)

大规模 Claude、MolClaw/MCP 与 GPU 训练都不是默认测试。历史资产迁移必须写入新目录，不能覆盖原结果。
