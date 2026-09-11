# Drug-Pipe 测评端故障分析报告

报告日期：2026-09-11。范围：2026-09-09 三组 MS-3/MO 测评、旧 SFT 的 MS-1/MS-2、2026-09-10 低学习率 SFT 五任务测评，以及截至 2026-09-11 14:57 的串行 MS-3 阶段性结果。本文复核已有文件，不持续监控运行中的任务，不修改模型回答、不启动额外测评。

## 1. 结论与归属原则

本轮问题不能统一归为“模型小”或“MCP server 过载”。已确认客户端本地 shell 缺少可用沙箱运行产物、整题级故障统计漏记工具错误、评分规则实施偏差，以及部分模型传参与文件交接错误。远端返回则确认存在 QuickVina 执行失败、Boltz 缺少预期产物、输入校验拒绝等情况。

`Error: fetch failed` 的直接证据是“某次具体工具调用未取得正常结果”。目前不能根据这句话确定失败发生在本机、SSH 反向隧道、代理、MCP 网关、工具服务器，还是服务器访问外部依赖时。客户端 SDK 异常和服务器返回的错误都可能被呈现为工具错误文本，不能从显示文本倒推出唯一来源。

本报告区分：

- **客户端确认问题**：本机代码、运行环境、模型决策或实验配置有直接证据。
- **服务端执行失败已确认、根因待查**：服务器返回了计算阶段、退出信息或产物诊断；不自动等同于服务端实现缺陷。
- **服务端正常拒绝错误输入**：错误在远端报告，但责任可能在客户端输入，不列作服务器宕机。
- **调用链归属未定**：缺少网络异常原因和服务端请求对应记录，不强行二分责任。

“题目出现过工具错误”不等于“题目始终失败”；同题可能先失败再成功。以下数字不是故障率，除非明确给出相应分母。调用次数包含重复尝试。

## 2. 调用链与取证边界

实际部署链路为：模型生成工具调用 → DSH 工具执行层 → MCP SDK Streamable HTTP → worker 上的代理配置/SSH 反向隧道 → 登录节点代理 → SCP MCP 网关/工具服务 → 计算程序、文件系统或外部数据库。

本地 `bash` 则走 DSH 本地 shell 与沙箱，不经过 MCP server。

部署代码中 worker 使用 `NODE_USE_ENV_PROXY=1`、指向 `127.0.0.1:13208` 的 HTTP(S) 代理；登录端脚本维持 SSH 反向转发。它们是有证据的链路组件，不是已经证实的故障原因。共享代理端口也不等于端口冲突：各 worker 的 loopback 网络独立。

已读取：最终回答、原始 diagnostic_transcript、工具调用参数与返回、评分文件、训练/评测 manifest、DSH bridge 和 runner 源码。未读取：MCP server 实时进程、完整服务端 stdout/stderr 文件、网关请求日志、历史 GPU/队列/连接池负载。服务端返回中的日志路径是定位线索，不代表已经打开过那些文件。

## 3. 客户端问题及原因分析

### C1. 本地 bash 沙箱不可用——已确认，运行产物已补齐

涉及工具：`bash`，不是 MolClaw 工具。原版 L1 的 MS3-007，事件 seq=1732，在请求 `cat /tmp/rank_output.json` 时返回：

```text
Error: sandbox mode "workspace-write" is requested but no sandbox backend is usable on this host; refusing to run the command unconfined.
```

命令在执行前被阻止。历史三组 MS-3 中，原版 L1 24/25 题、原版层级技能 23/25 题、旧 SFT 16/25 题出现此错误。它妨碍 Python 数据处理、文件转换、结果整理及本地检查，但不表示所有文件读写工具也失败。

本地检查发现 Landlock 平台包对应的可执行文件缺失。已从现有 C 源码构建静态 musl 运行产物；新 worker 内核探针显示 partial enforcement（旧 ABI），实际“工作目录内允许写入、目录外拒绝写入”测试通过。没有通过关闭沙箱绕过故障。登录节点自身不支持 Landlock 不影响 worker 的独立实测结论。

**原因边界**：可以确认新 worker 的该路径恢复；历史节点没有逐一现场复现，不能断言所有旧节点都仅缺同一个文件。二进制是运行产物，不能只依赖 git checkout；需在部署准备中构建/安装并作一次真实 shell 探针。已有 tools/list 和模型 tool-call 探针不能代替这一步。

### C2. 整题级故障判定漏记工具级错误——已确认，尚未完整修复

`run_dsh_molbench.py` 的 `failure_class()` 对 `status != failed` 直接返回 None。模型遇到工具失败后仍给出最终回答，整题便可能是 completed。`infra_failure_count` 又只统计 `failure_class == retryable_infra`，所以不能代表轨迹内工具健康。

第二层漏记来自响应语义：本报告从历史三组 MS-3 中抽取到 232 条 JSON 正文明确带 `status=error/failed/false` 的结果，外层 `isError` 全为 false。该数字是这个筛选条件下的结果条数，不是全部工具错误数。正文报错与 MCP 外层标志不一致，会让只看外层的统计漏掉业务失败；也不能把这些 false 当作科学计算成功。

DSH MCP bridge 只在 MCP `result.isError === true` 时转为异常，不自动理解每种工具自己的 `status`。这与服务器选择将业务状态放在正文的接口设计有关，不宜把所有非 success 状态一律升级为可重试基础设施故障。

**影响**：之前“所有 rjob 成功退出、infra_failure_count=0，所以没有未解决基础设施问题”的汇报不成立。`publishable=true` 目前只说明通过现有门控，不能保证真实工具环境完整。历史分数应保留，同时附加工具故障说明。

**建议**：轻量分开记录 task 结束原因、传输错误、本地环境错误、工具业务错误；保留原始返回。输入错误不自动重试，科学错误不重跑挑最好答案。不要为此添加大量新的样本拒绝门槛。

### C3. 错误工具名、数据库 ID 与文件交接——模型执行问题

原版 L1 MS3-007：

- `retrieve_protein_structure_by_pdb_id` 曾收到 `{"pdb_id":"CHEMBL4722"}`，把 ChEMBL 标识符当作 PDB ID。
- 模型曾调用不存在的 `mcp__molclaw-scp__quickvina-docking`，返回 unknown tool。这不是远端计算崩溃。
- 在没有取得可靠结构时，模型提交 `/tmp/aura_kinase.pdb` 与口袋坐标，之后又用本地 write 创建简化 PDB。`fpocket_toolkit` 返回 `pdb_file not found: /tmp/aura_kinase.pdb`（seq=1105）。本地文件不自动存在于远端服务，轨迹未完成正确上传。
- 同题 `is_valid_smiles` 对三个分子成功返回（seq=992），证明不能概括为整题所有远端工具都不可用。

原版层级技能 MS3-018 的 `karmadock_tool` 把同一个 `.pdb` 路径同时作为 ligand_smi、protein_file、crystal_ligand_file，且 dry_run=true。服务器返回 `invalid_coordinate_input` / `MOL2 contains no atom coordinates`。这个案例优先归为输入类型/工作流误用，而非 KarmaDock 服务器不可用。

**建议**：工具名和参数按现有 schema；对文件保持“本地文件/服务器文件”区分，沿用工具返回路径和上传下载工具；不要自行推测远端路径。本文不将候选数量、重复或候选外字符串设为评分拒绝条件，也不再针对这些问题追加门控。

### C4. 模型输出超限与缺乏观察支持的回答——模型侧现象，成因未完全分离

低学习率五任务共 190 题，22 题以 model_max_tokens 失败：MS-1 2、MS-2 4、MS-3 5、MO-Opt 3、MO-Edit 8。它表示触及单次模型输出上限，不是 GPU OOM，也不能自动判为整题总上下文耗尽。

工具连续报错可能促使反复尝试、长篇解释，但没有逐请求 token 分解或控制实验，不能断言它是所有超限的原因。原版 L1 MS3-007 用 QED/结构经验替代实际 docking；这不能声称是完成了要求的 docking 科学流程。环境失败和模型随后作出的不可靠替代决策可以同时存在。

### C5. 评分规则与实施偏差——已按用户要求修正

初始 MS-3 评分要求完整 60 候选排列，导致部分已有列表无法进入科学评分。用户已明确要求取消数量、重复、候选归属门控。

首次修改又额外取消 evidence 和精确字段检查，超出了授权；随后口头承诺恢复但未立即落实，是客户端实施和交付检查的问题。2026-09-11 已恢复，当前 `top3_list_v2`：

```json
{"ranked_smiles":["SMILES_1","SMILES_2","SMILES_3"],"evidence":[]}
```

保持原始字段和类型要求；不检查列表长度、重复、候选归属；原列表前 3 项原位进入上游评分，不去重、不删除、不补位。正文/代码块提取仍按既有三个独立策略统计。上游 hit_at_3 是前 3 项至少一个命中 GT answer 集合的题目比例；top3_hit_rate 是命中位置数/3 的平均。重复命中按上游位置计数保留，这是用户接受原列表、保留上游数学定义后的行为。

5 个相关回归测试通过。新旧 SFT 的 MS-3 在独立目录统一重算，保留历史评分文件。旧 SFT 1/25 题命中，新 SFT 2/25 题命中；不能把规则改变导致的变化记为模型能力提升。没有修改训练端校验或悄悄修补模型回答。

### C6. 配置差异限制学习率归因

旧 SFT MS-1/MS-2 为 TP4，新低学习率测评为 TP2；旧环境存在本地沙箱故障，新环境已恢复。旧扩展任务三模型重叠运行，新串行 MS-3 每次仅一模型、题目并发 1；低学习率运行仍是题目并发 2。变化不止学习率。

低学习率 MS-2 为 12/37，旧 SFT 为 22/37；新一轮有 22 次 fetch failed，覆盖 13/37 题，其中 Boltz2 21 次、序列检索 1 次；旧 MS-2 未记录到 fetch failed，但有 bash 沙箱错误。可以报告实测分数下降，不能据此证明“学习率降低导致科学能力下降”。

## 4. MCP server 侧现象及原因分析

以下“服务端执行失败”指服务器返回中记录了失败阶段，并非已完成服务端现场根因复现。

### S1. QuickVina 子进程断言失败

工具：`mcp__molclaw-scp__molecule_docking_quickvina_fullprocess`。原版层级技能 MS3-003，seq=2431，request_id=`26f00638c63044b4a4fc3a946d1d6ca5`。

```text
返回码: -6
QuickVina2-GPU: ./lib/parse_pdbqt.cpp:71:
Assertion `i-1 < str.size()' failed.
```

返回 diagnostics.error_code=quickvina_execution_failed、stage=quickvina，并给出 stdout/stderr 路径。这是收到服务器结果后的计算程序失败，区别于裸 fetch failed。该例 diagnostics 没有报告网格越界，不能无证据归为 GPU 资源不足或口袋网格超过限制。

模型传入的 receptor 路径后缀为 `.pdbqt`；需要核对该工具的输入约定以及具体 receptor/ligand 文件。根因可能是输入格式、转换输出或解析器边界情况，现有证据不足以确定是哪一个。错误正文包含退出码 -6，但 diagnostics.return_code 为 null，诊断字段本身也不完整。

服务端排查入口：请求目录 `molecule_docking_quickvina_fullprocess_20260909_134147_26f00638c630` 下的 stdout.log/stderr.log，以及触发断言的实际 PDBQT。只重放这个请求即可，没必要先重跑整个 benchmark。

### S2. QuickVina 配体 PDBQT 解析失败

同一工具，原版 L1 MS3-002，seq=2489，request_id=`6602ecfc770a4a46b1ee2c5a9455540b`。

```text
返回码: 1
Parse error on line 85 ... smiles_0_0909_131700539.pdbqt:
Unknown or inappropriate tag
```

输入 SMILES 含点分隔组分；其他含 Br. 或 [I-] 的记录也出现解析报错。这支持检查多组分分子的转换/对接兼容性，但不证明所有盐类必然失败，或该错误只由盐导致。

该例请求 pocket size 为 20/15/18，返回 diagnostics 的 sizes_angstrom 是 25/25/25。已确认请求值和诊断值不一致，尚不清楚是服务端调整为实际值、诊断写了默认值还是其他原因。需要记录实际执行参数；不能只依据客户端配置声称 docking 参数完全按请求生效。

不要为提高分数而默认去盐、删候选或改写 SMILES。若工具本身不支持某类合法输入，服务端应明确能力限制、提供可解释错误；科学预处理变更必须对所有模型一致。

### S3. Boltz 正常退出但缺少亲和力产物

工具：`mcp__molclaw-scp__pred_binding_affinity_boltz2`。返回 error_code=`boltz_predictions_missing`、stage=`output_validation`、return_code=0；affinity_json_count=0，cif_count=1。

这说明服务端检查到了“有结构输出，但没有预期的亲和力 JSON”。退出码 0 不足以说明任务成功；现有服务端产物校验在这个案例起到了作用。

具体根因仍需查看保留的 stdout/stderr、输入 YAML、启用的 affinity 配置和实际结果目录。返回日志提到外部 MSA 服务，但没有证据证明此处就是 MSA 服务故障；也不能归为本机 JSON 格式问题。

### S4. 服务端业务错误与 MCP 外层状态不一致

232 条被抽取的结构化业务错误外层 isError=false，包括找不到文件、QuickVina 崩溃、Boltz 输入拒绝等。它们不能一律叫服务器故障，其中有正常输入校验；但“调用传输成功”和“业务计算成功”缺少统一易消费的区分。

建议服务端/桥接端约定稳定的业务状态及错误类别，保留 error_code、stage、request_id、实际执行参数、日志路径；客户端据此统计。不是简单把所有 status=false 都重试，也不假设所有工具必须共享同一内部 JSON schema。

### S5. 服务端报告错误，不等于服务器有缺陷

`fpocket_toolkit` 找不到本地 /tmp 路径、`karmadock_tool` 收到错误文件类型、`prepare_protein_md` 拒绝含未处理 HETATM 的结构，均可能是正常拒绝。`retrieve_protein_structure_by_gene_name` “未找到结构”可能来自标识符不合要求、数据库覆盖或外部检索失败；不能仅凭文字判定服务器不可用。

历史回答中“Boltz 128 原子限制”等描述有部分来自模型解释。未用明确工具诊断核实前，不列作已证实服务限制；不能把模型的故障解释当作服务器日志。

## 5. fetch failed：共享链路问题与并发假设

详细工具计数见附录。历史三组 MS-3 分别 417 / 540 / 304 次，涉及 19 / 22 / 14 道题。多个工具共享 MCP 连接路径，同一工具同时存在成功和失败，不能据此判断每个科学工具分别坏了。

“多个模型并行给服务器施压”是合理待验假设。客户端可能同时发出多条工具调用；即使题目并发为 1，也不保证单个 agent 的多个工具调用串行，更不保证平台没有其他用户。因此现有串行队列降低的是本项目模型与题目级并发，不是服务器全局并发锁。

已执行的措施：低学习率运行结束后，原版 L1 → 原版层级技能 → 旧 SFT L1 的 MS-3 依次提交，每个 2 卡、题目并发 1，由 tmux 队列等前一任务结束再启动下一任务。不是提前提交三个 rjob 给调度器同时排队。

截至此前 2026-09-11 14:57 的检查，串行原版 L1 完成 16/25 题，保存的轨迹仍有 57 次 fetch failed，涉及 10 题。这不能与完整旧运行直接比较成失败率，却足以说明“改成模型串行后错误完全消失”尚不成立。此报告没有新增实时监控。

需要区分的候选原因：

| 位置 | 可能原因（未确认） | 最小需要的证据 |
|---|---|---|
| 客户端/SDK | HTTP 连接、请求中止、超时、连接池问题 | 异常类型、cause.code、耗时、请求 ID |
| 隧道/代理 | 隧道短暂断连、代理连接失败 | 同时段 tunnel/代理日志、请求时间 |
| MCP 网关 | 限流、连接/会话异常、上游超时 | HTTP 状态、网关 trace ID、网关日志 |
| 计算服务器 | 并发排队、子进程失败、资源不足 | 对应请求的队列等待、运行日志、资源曲线 |
| 外部依赖 | 结构数据库、MSA 或其他下载失败 | 服务端出站错误、目标服务和状态 |

最小取证方式：下次错误保留工具名、脱敏参数、callId、起止时间、完整异常 cause/状态码；有服务器 request_id 时与服务端日志关联。选择一条已失败且可定位的请求做单次复现。没有关联记录前，不给“服务器过载”下结论。

## 6. 已处理、待处理与对实验结论的影响

| 项目 | 当前状态 | 后续动作 |
|---|---|---|
| 本地沙箱 | 新 worker 已补运行产物并功能验证 | 固化部署准备与一次启动探针 |
| MS-3 完整排列门控 | 已取消 | 保持用户指定 Top-3 规则，不额外审查候选排列 |
| JSON 字段意外放宽 | 已恢复原约定，5 个测试通过 | 使用 v2 评分说明，历史结果独立重算 |
| 项目模型并行 | 已改 tmux 串行队列 | 等完整结果再判断压力假设，不持续人工轮询 |
| 工具级故障漏记 | 已定位，尚未全面改实现 | 小范围增加可区分错误统计，避免复杂拒绝规则 |
| 远端 fetch 根因 | 未确认 | 同一请求客户端—网关—服务端关联取证 |
| QuickVina/Boltz 执行异常 | 已有返回证据，未现场复现 | 按附录 request_id 检查保留日志及输入 |
| 新旧学习率比较 | 只能描述实测，不能纯因果归因 | 说明 TP、沙箱、并发和调用失败差异 |

工具故障未从科学评分分母剔除，未通过挑选成功重跑提高成绩。报告中的环境诊断用于解释结果局限，不用于回填科学答案。新规则的 MS-3 1/25 与 2/25 差一题，不足以支持稳定模型提升；低学习率 MS-2 下降也不能直接排除工具故障影响。

## 7. 证据索引与复核方法

- `structured_tool_errors.json`：历史三组 MS-3 的 232 条结构化业务错误。每条包含 run、task、event seq、工具、调用参数、完整业务返回、外层 isError、原始轨迹路径。筛选依据为正文可解析为 JSON 对象且显式 status 为 error/failed/false；不包含所有文本异常或嵌套单分子失败。
- `fetch_failed_by_tool.json`：历史三组逐工具 fetch failed 次数、受影响题数，及 MS3-007 具体调用。依据工具/result 文本，不依据模型复述；按 callId 绑定工具名。
- 原始运行目录：`slime-wd/outputs/dsh_molbench_evals/aligned-v4-9b-{orig-l1,orig-hier,sft-l1}-ms3-mo-full-0909c/`。
- 新模型：`slime-wd/outputs/dsh_molbench_evals/aligned-v4-9b-lr2e6-l1-all5-0910b/`。
- 沙箱证据：`slime-wd/outputs/infra-aligned-v4-9b-lr2e6-l1-all5-0910b/sandbox_preflight.log`、`sandbox_behavior_test.log`、`sandbox_binary.sha256`。
- 比较结果：共享发布目录 `drug_pipe_regular_v1_20260908/experiments/lr_comparison_20260911/`；队列配置：`experiments/ms3_serial_top3_0910a/`。
- 客户端代码：`slime-wd/dsh-molbench/run_dsh_molbench.py` 的 failure_class、publishable_records、project_prediction；`pretrained_matrix/run_worker.sh` 和 submit_one.sh 的代理、隧道、并发设置。
- MCP bridge：`slime-wd/deepseek-harness/packages/mcp/mcp-client/src/tools.ts` 的 callToolUncached/create executor；transport.ts 的 Streamable HTTP 创建逻辑；`packages/core/tools/src/index.ts` 的错误文本呈现逻辑。

附录 JSON 中保留的是当次返回信息。内含远端日志路径并不代表本地存在这些文件。结论按照证据强度标注；未读取的服务端日志不写成已验证原因。

## 附录 A：主要工具的 fetch failed 统计

每格为报错调用次数 / 涉及题数。工具均有 `mcp__molclaw-scp__` 前缀。

| 工具 | 原版 L1 | 原版层级 | 旧 SFT L1 |
|---|---:|---:|---:|
| `retrieve_protein_structure_by_pdb_id` | 47 / 10 | 35 / 12 | 20 / 5 |
| `retrieve_protein_structure_by_gene_name` | 24 / 10 | 39 / 14 | 28 / 7 |
| `retrieve_protein_structure_by_uniprot_id` | 9 / 7 | 8 / 6 | 11 / 7 |
| `retrieve_protein_sequence` | 53 / 10 | 39 / 9 | 2 / 2 |
| `fix_pdb` | 23 / 9 | 20 / 8 | 23 / 10 |
| `fpocket_toolkit` | 23 / 10 | 18 / 7 | 26 / 6 |
| `molecule_docking_quickvina_fullprocess` | 38 / 10 | 87 / 13 | 79 / 6 |
| `pred_binding_affinity_boltz2` | 89 / 17 | 119 / 17 | 8 / 2 |
| `karmadock_tool` | 5 / 5 | 4 / 4 | 12 / 4 |
| `server_file_to_base64` | 3 / 3 | 6 / 5 | 14 / 6 |
| `calculate_mol_basic_info` | 16 / 7 | 21 / 9 | 9 / 5 |
| `is_valid_smiles` | 6 / 4 | 50 / 12 | 14 / 8 |

## 附录 B：服务端请求定位

下列日志目录来自服务器返回；本次没有读取其内容。

- orig-hier / molbench_ms3_003 / seq=2431：`mcp__molclaw-scp__molecule_docking_quickvina_fullprocess`
  - request_id：`26f00638c63044b4a4fc3a946d1d6ca5`
  - stdout：`/data/lwj/wll/code/DrugAgentTools/sxy_local/tool_result/quickvina_result/molecule_docking_quickvina_fullprocess_20260909_134147_26f00638c630/stdout.log`
  - stderr：`/data/lwj/wll/code/DrugAgentTools/sxy_local/tool_result/quickvina_result/molecule_docking_quickvina_fullprocess_20260909_134147_26f00638c630/stderr.log`
- orig-l1 / molbench_ms3_002 / seq=2489：`mcp__molclaw-scp__molecule_docking_quickvina_fullprocess`
  - request_id：`6602ecfc770a4a46b1ee2c5a9455540b`
  - stdout：`/data/lwj/wll/code/DrugAgentTools/sxy_local/tool_result/quickvina_result/molecule_docking_quickvina_fullprocess_20260909_131700_6602ecfc770a/stdout.log`
  - stderr：`/data/lwj/wll/code/DrugAgentTools/sxy_local/tool_result/quickvina_result/molecule_docking_quickvina_fullprocess_20260909_131700_6602ecfc770a/stderr.log`
- orig-l1 / molbench_ms3_017 / seq=2526：`mcp__molclaw-scp__pred_binding_affinity_boltz2`
  - request_id：`646d289697eb4e31b12723e71b21617e`
  - stdout：`/data/lwj/wll/code/DrugAgentTools/sxy_local/tool_result/boltz_affinity_result/pred_binding_affinity_boltz2_20260909_171910_646d289697eb/stdout.log`
  - stderr：`/data/lwj/wll/code/DrugAgentTools/sxy_local/tool_result/boltz_affinity_result/pred_binding_affinity_boltz2_20260909_171910_646d289697eb/stderr.log`
- orig-hier / molbench_ms3_018 / seq=3154：`mcp__molclaw-scp__karmadock_tool`
  - request_id：`c6ac67ffde734d529105fb7e39ef6e1a`
  - stdout：`/data/lwj/wll/code/DrugAgentTools/sxy_local/tool_result/karmadock_tool_result/karmadock_tool_20260909_154448_c6ac67ffde73/stdout.log`
  - stderr：`/data/lwj/wll/code/DrugAgentTools/sxy_local/tool_result/karmadock_tool_result/karmadock_tool_20260909_154448_c6ac67ffde73/stderr.log`
