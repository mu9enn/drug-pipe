# Known Issues

- Data-Pipe 产物目录与 Slime 数据目录相互独立；需要把 canonical `react_trajectories.jsonl` 放入 `$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl`，或显式传入 `PROMPT_DATA`/`INPUT`。
- Slime launcher 仍通过外部 `slime_env.sh` 定位 workspace、模型和 CUDA 环境。换机器时应显式设置 `SLIME_ENV`、`SLIME`、`DATA`、`DRUG_AGENT_DATA_ROOT` 与 `DRUG_AGENT_RUNS_ROOT`。
- Qwen3.5-4B 长 ReAct 已配置 TP=4/DP=1、batch divisibility、tail、resume 与 `expandable_segments` guard，但本次重构没有启动 GPU smoke/full training。
- 正式训练允许本地 Ray/SGLang/GAD discriminator 网络流量；静态审计只能证明代码引用边界，不能代替运行机网络策略。
- evaluator 对 VS/AC/PF 强制要求 RDKit；缺失时明确失败并拒绝对应样本，不会退化为 raw string equality。KG/E2E 不依赖 RDKit。
- ToolRL allowlist 与 GAD 的方法筛选属于训练策略 projection，可能随真实 MolClaw tool inventory 演化；更新它们不能反向改变 Tool Catalog 或 canonical KG。
- 历史 KG 迁移发现 graph/scored projection 与原 adjudication 有冲突时，以原 Claude adjudication 为准并写 conflict report；不能把 projection 自动提升为新语义。
- Tool-KG 的 ontology、taxonomy 与 sampling profile 已改为各自单一控制面，但本轮没有用真实 Claude 重跑 Tool Card/pair adjudication/Stage3；首次正式重跑仍需检查 annotation evidence、ontology rejection rate 与 sampling manifest。
- 正式清洗固定为 `python_clean → llm_clean`。Python drafts 不是 accepted 数据；LLM patch
  缺失、超时、JSON/schema 非法或越权修改都会进入 quarantine，并在 `debug/` 保留
  stdout/stderr 与隔离 workdir。
- 本轮没有重新运行大规模 Claude/MolClaw、没有批量 semantic repair，也没有验证远端 MCP 服务可用性。
