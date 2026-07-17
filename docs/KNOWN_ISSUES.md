# Known Issues

- SFT、ToolRL、GAD 的部分 launcher 默认仍指向 owner 运行过的 `mcp_sft_all*` 路径。为避免未经确认改变默认命令，当前 canonical 主线需要显式传 `PROMPT_DATA`/`INPUT`；确认后可统一默认文件名。
- Slime 环境脚本仍兼容 `/root/...` 与 `/home/sunxiangyu/...` 两套绝对工作区路径。换机器时应显式设置 `SLIME_ENV`、`SLIME`、`DATA`、`DRUG_AGENT_DATA_ROOT` 与 `DRUG_AGENT_RUNS_ROOT`。
- Qwen3.5-4B 长 ReAct 已配置 TP=4/DP=1、batch divisibility、tail、resume 与 `expandable_segments` guard，但本次重构没有启动 GPU smoke/full training。
- 正式训练允许本地 Ray/SGLang/GAD discriminator 网络流量；静态审计只能证明代码引用边界，不能代替运行机网络策略。
- evaluator 在 RDKit 不可用时不能做 SMILES canonicalization；curation report 会记录 `chemistry_available=false`，此时应先补环境再比较科学指标。
- ToolRL allowlist 与 GAD 的方法筛选属于训练策略 projection，可能随真实 MolClaw tool inventory 演化；更新它们不能反向改变 Tool Catalog 或 canonical KG。
- 历史 KG 迁移发现 graph/scored projection 与原 adjudication 有冲突时，以原 Claude adjudication 为准并写 conflict report；不能把 projection 自动提升为新语义。
- 本轮没有重新运行大规模 Claude/MolClaw、没有批量 semantic repair，也没有验证远端 MCP 服务可用性。
