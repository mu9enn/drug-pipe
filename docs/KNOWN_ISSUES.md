# Known Issues

- Data-Pipe 产物目录与 Slime 数据目录相互独立；SFT/ToolRL launcher 应显式把 `PROMPT_DATA` 指向相应的 `qwen35_sft.jsonl` / structured ToolRL view。
- 旧 v7/v8 tool catalog 的 parameter descriptions 仍可能写有 `resource://` path contract；新的 SFT
  adapter 会拒绝这种 manifest。正式 materialization 必须使用按当前 deployment 原生 schemas 重新导出的
  clean manifest，不能把旧 path annotation 带回训练正文。
- Slime launcher 仍通过外部 `slime_env.sh` 定位 workspace、模型和 CUDA 环境。换机器时应显式设置 `SLIME_ENV`、`SLIME`、`DATA`、`DRUG_AGENT_DATA_ROOT` 与 `DRUG_AGENT_RUNS_ROOT`。
- Qwen3.5 structured messages 已通过真实 tokenizer 与 `qwen3_5` loss-mask 单样本验证，但本次重构没有启动 GPU smoke/full training。
- 正式训练允许本地 Ray/SGLang/GAD discriminator 网络流量；静态审计只能证明代码引用边界，不能代替运行机网络策略。
- evaluator 对 VS/AC/PF 的化学指标要求 RDKit；缺失时明确记录 evaluator error，不会退化为 raw string equality，也不会单独拒绝清洗样本。KG/E2E 不依赖 RDKit。
- ToolRL allowlist 与 GAD 的方法筛选属于训练策略 projection，可能随真实 MolClaw tool inventory 演化；更新它们不能反向改变 Tool Catalog 或 canonical KG。
- 历史 KG 迁移发现 graph/scored projection 与原 adjudication 有冲突时，以原 Claude adjudication 为准并写 conflict report；不能把 projection 自动提升为新语义。
- Tool-KG 的 ontology、taxonomy 与 sampling profile 已改为各自单一控制面，但本轮没有用真实 Claude 重跑 Tool Card/pair adjudication/Stage3；首次正式重跑仍需检查 annotation evidence、ontology rejection rate 与 sampling manifest。
- 正式 SFT 清洗顺序是 raw → pre-clean native audit → semantic/Python deterministic cleaning → mandatory LLM reasoning
  patch → Qwen3.5 SFT adapter。LLM patch 缺失、超时、schema 非法或越权不会删除 semantic
  母数据；该条只进入 `llm_pending.jsonl`，不得作为成功 cleaned materialization。
- 旧 ToolRL/online runtime 仍有 XML/ReAct 依赖；它们被明确隔离在后续 ToolRL 专项范围，正式 SFT
  materializer 和文档命令不会调用这些入口。
- 登录节点当前没有安装 SGLang，因此 native parser 的真实 round-trip gate 必须在实际训练 worker
  上用该 worker 的 checkpoint/SGLang 版本运行；launcher 在 Ray/GPU 启动前会强制执行此 gate。
- 本轮没有重新运行大规模 Claude/MolClaw、没有批量 semantic repair，也没有验证远端 MCP 服务可用性。
