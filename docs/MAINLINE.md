# Mainline

## 职责与唯一 authority

| 事实 | 唯一 authority | 其他模块的角色 |
| --- | --- | --- |
| MCP 明确的工具字段 | MCP schema 的确定性 snapshot | Tool Catalog 消费 |
| skills 中的工具语义摘要 | Tool-card Claude agent | Python heuristic 仅提示缺失信息 |
| 是否调度 directed candidate | stage taxonomy 的 transition/alternative 规则 | Tool Card 字段只用于上下文、优先级与 audit |
| directed relation status/type/mapping/evidence/confidence | Claude pair adjudication | canonical parser 只校验并写 `edge_decisions.jsonl` |
| sampling graph | `graph.jsonl` 的确定性 projection | 只筛选可采样的 valid Claude decision，不改边语义 |
| grounded task validity | Tool-KG sampler + Science-KB + canonical graph | Data-Pipe 只检查可读性并执行 |
| task metrics / `task_answer_valid` | `pipeline/evaluate/task_evaluator.py` | curator 消费结果 |
| `execution_valid`、`training_trace_valid`、MolClaw usage | `trace_curator.py` | acceptance gate 消费 |
| canonical ReAct construction | `react_constructor.py` | LLM/hard clean 只受保护地清理 |
| accepted/rejected/quarantine | `cleaning/acceptance_gate.py` | evaluator、LLM clean、hard clean 只提供事实/findings |
| decision-state slicing / assistant decision parsing | `drug_agent/decision_extractor.py` | ToolRL、GAD 只派生方法字段 |
| ToolRL reward | `drug_agent.toolrl.molclaw_reward` | 只评价生成 action，不执行 |
| GAD negative/discriminator/reward | `drug_agent.gad` | 保留现有方法与权重 |
| real MCP execution | Data-Pipe executor 与显式 `tools_debug` | formal training 禁止调用 |

ToolRL 的 allowlist、GAD 的筛选策略和 tool registry 是方法策略或运行时 projection，不是新的工具语义 authority。

## 主线图

```mermaid
flowchart TD
    A[MCP schema + canonical skills] --> B[Tool Catalog]
    B --> C[Taxonomy-directed pair scheduling]
    C --> D[Claude directional adjudication]
    D --> E[Canonical edge decisions]
    E --> F[Pure sampling graph]
    F --> G[Grounded task sampling]
    G --> H[Real Agent execution]
    H --> I[Raw complete_session]
    I --> J[Task evaluator + ReAct construction]
    J --> K[LLM clean + hard clean + final gate]
    K --> R[Canonical ReAct + audit sidecar]
    R --> L[Shared history-only decisions]
    L --> M[SFT]
    L --> N[ToolRL]
    L --> O[GAD]
    M --> P[Checkpoint]
    N --> P
    O --> P
    P --> Q[Explicit online MCP evaluation/debug]
```

## Offline / online 边界

SFT 使用完整历史 teacher forcing；ToolRL 和 GAD 在固定历史 state 上生成下一步 action。生成 action 不会被执行，也不会取得新 observation。正式训练入口在启动 Ray 前加载 `offline_training_env.sh`，清除 MolClaw credentials，并由 MCP client/executor fail closed。

真实工具交互只允许在 Data-Pipe 执行层和显式 online debug 中发生。`debug_mcp_tools.py`、`debug_one_task.py` 与 `debug_replay_trajectory.py` 必须设置 `DRUG_AGENT_ALLOW_TOOL_ENV=1`。

## 兼容层

- Tool-KG `score` 命令委托 canonical edge builder；旧 graph views/CSV/GraphML 只按需导出。
- `trajectory_exporter.py`、`scan_molclaw_usage.py`、`post_process_sft.py` 保留旧入口形状，但 authority 都在 evaluator/curator。
- SFT、ToolRL、GAD 与 online replay 的默认输入都从 `$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl` 或其方法派生目录开始；`PROMPT_DATA`/`INPUT` 只用于显式覆盖。

OPD、VERL bundle、legacy action-JSON SFT 和 legacy online PPO/GRPO 不属于当前主线。
