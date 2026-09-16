# Fetch failed 根因调查：交给可访问两端机器的 Codex

更新：2026-09-11。本文独立提供上下文、已验证证据、入口、复现实验和下一步任务，不要求接手者阅读聊天记录。

## 0. 接手目标与当前结论

项目 Drug-Pipe 用 DSH（deepseek-harness）调用 MolClaw 工具评测药物任务。GPU worker 不能直接联网，通过 SSH 与 CPU 登录节点中转出网，历史测评频繁出现 `Error: fetch failed`。另一台 CPU 机器通过模型 API 采集的 200 条轨迹，用户报告 4278 次 MolClaw 调用没有出现该字符串。用户将调查交给笔记本 Codex，以便同时访问本项目机器和另一台 API 采集机器。

**目标：确定失败来自哪一层、以什么机制发生，并用最小对照验证修复；不是再次统计字符串，也不是无限重试掩盖故障。**

截至本次交接：

- 已查清实际网络链路和关键代码。客户端只保留异常 message，丢失底层 cause/code，这是已确认的取证缺陷。
- 同一 GPU 转发链路上的轻量工具和一个历史 PDB 案例现在成功；本次短探测没有复现真实 MCP `fetch failed`。
- 发现 MCP 四小时请求预算与 Node HTTP 默认 300 秒响应头/响应体空闲超时分层独立，**这是待验证原因，不是已经证明的历史根因**。
- 已新增被动 HTTP 诊断器，真实本地断连测试可记录 `UND_ERR_SOCKET`。**本地模拟不是 MCP 真实故障证据。**
- 还没有历史失败的底层错误码、同一请求的网关/服务端对应日志，也未读取另一台采集机器的配置。**根因尚未闭环；不能声称已证明 SSH 有问题或 MCP server 无关。**

用户偏好：直击核心、最小改动，不要增加庞大审计/重试框架，不做无关训练修改。不要把单次出现工具错误等同于最终答错。当前任务是调查交接；没有新授权停止正式作业、重启共享 relay/MCP 服务或迁移正式评测架构。

## 1. 先区分四类机器

| 角色 | 已知作用 | 本次调查可见性 |
|---|---|---|
| 本项目 CPU 登录节点 | 存放仓库；运行 relay、SSH 转发控制器、tmux 队列 | 已读取进程、代码和日志 |
| 本项目 GPU worker | 部署本地 Qwen/SFT 模型；运行 DSH、Node MCP SDK | 曾 SSH 访问当前 worker，完成只读探测 |
| 用户另一台 CPU/API 采集机器 | 跑 200 条轨迹；模型通过 API 调用 | 原调查者不能访问，待笔记本 Codex 核查 |
| MCP 网关/实际工具服务器 | 网关收请求、工具执行计算与访问数据库 | 只看到客户端响应和响应中的远端路径，未读实际机器日志 |

“能访问另一台 API 采集机器”不自动意味着能访问 MCP 网关和工具服务主机。接手先核对实际权限与主机关系，不要把不同 CPU/server 当成同一台。

## 2. 路径、版本和配置

以下用 P 表示项目根，W 表示 W=P/slime-wd：

```text
P（登录节点）=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe
P（物理共享路径）=/mnt/shared-storage-user/sdpdev-fs/sunxiangyu/drug-pipe
P（GPU worker）=/root/slime_sxy/group-space/sunxiangyu/drug-pipe
W=P/slime-wd
E（实验目录）=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug_wd/drug_pipe_regular_v1_20260908/experiments
```

MCP endpoint：`https://scp.intern-ai.org.cn/api/v1/mcp/2/DrugSDA-Tool`。

认证 Header 名称 `SCP-HUB-API-KEY`，值来自环境变量 `MOLCLAW_SCP_API_KEY`。本机凭据文件 `/home/sunxiangyu/.dsh/molclaw.env`，worker `/root/.dsh/molclaw.env`。**不要输出/复制凭据正文到交接文档、日志或 git；探测脚本在原机器本地读取。**

当前实测运行时：Node **v24.19.0**；Node 内置 Undici **7.29.0**；MCP SDK **1.29.0**。依赖树另有独立包 Undici **7.28.0**，不要把它误当生产内置 fetch 的版本。本地断连 fixture 使用该独立包创建 dispatcher，但 fetch 是实际 Node 内置 fetch。

Node 二进制：`W/outputs/dsh_eval_runtime/node-v24.19.0/bin/node`。

正式 MCP 配置 `W/dsh-molbench/pretrained_matrix/molclaw.cordis.patch.yml`：

```yaml
serverName: molclaw-scp
transport: streamable-http
url: https://scp.intern-ai.org.cn/api/v1/mcp/2/DrugSDA-Tool
headers:
  SCP-HUB-API-KEY: !!js process.env.MOLCLAW_SCP_API_KEY
toolCallTimeoutMs: 14400000
failOnStartupError: true
```

模型环境作为背景：9B 原版 L1、9B 原版层级技能、旧 SFT L1、小学习率 2e-6 SFT L1；当前 worker 使用两张 H200、TP2，native thinking，greedy，262144 上下文，16384 单次输出上限。不要为了网络调查改变这些变量，也不需要加载模型就能做 MCP 隔离探测。

## 3. 实际链路：关键认知纠正

```text
GPU worker:
  DSH Node 进程 / MCP SDK / fetch
  HTTP(S)_PROXY=http://127.0.0.1:13208
       ↓ worker 本地监听端口
SSH 反向转发（由 CPU 登录节点发起 ssh -R）
       ↓ CPU 登录节点 127.0.0.1:13208
Python relay.py：仅复制双向 TCP 字节
       ↓
httpproxy-headless.kubebrain.svc.pjlab.local:3128
       ↓ 集群 HTTP 出口代理，HTTPS 隧道
scp.intern-ai.org.cn MCP 网关
       ↓
MolClaw 工具服务、计算程序、外部依赖
```

**MCP 客户端并未搬到 CPU。** 当前 CPU 只是 TCP 中转；MCP 的连接、HTTP 响应、SDK 流生命周期仍在 GPU worker 内。另一台“CPU 上跑 agent + API 调模型”方案可能不同，需要实测配置确认。

登录节点启动命令（现有服务，不要重复启动）：

```bash
MCP_RELAY_LISTEN_HOST=0.0.0.0 MCP_RELAY_LISTEN_PORT=13208 \
  bash "$P/slime-wd/molclaw-mcp-relay/run_relay.sh"
```

relay 默认目标 `httpproxy-headless.kubebrain.svc.pjlab.local:3128`。日志 `/home/sunxiangyu/mcp_http_proxy_relay.log`；tmux 名 `molclaw-relay-13208`。

转发实际选项：

```bash
ssh -NT -o BatchMode=yes -o ConnectTimeout=15 -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  -R 127.0.0.1:13208:127.0.0.1:13208 "$target"
```

GPU 环境有 `NODE_USE_ENV_PROXY=1`，大/小写 HTTP_PROXY、HTTPS_PROXY；NO_PROXY 为 `127.0.0.1,localhost`。模型本地 API 绕过出口代理。不同 worker 的 loopback 网络不同，不能仅因都使用 13208 就认定端口冲突。

## 4. 已核查源码和结论

路径均相对 P，行号以交接时工作树为准，接手可用函数名检索。

| 文件/位置 | 已核实内容 |
|---|---|
| `slime-wd/molclaw-mcp-relay/relay.py` | asyncio 双向 TCP 复制；没有 MCP 解析、模型调用或主动配置应用请求超时；EOF/异常会关闭对应 writer |
| `slime-wd/molclaw-mcp-relay/run_relay.sh` | relay 进程与日志路径 |
| `slime-wd/dsh-molbench/pretrained_matrix/submit_one.sh`，约 100–110 行 | SSH 反向转发及断开后循环重连；日志混合短 SSH 命令与长期转发 |
| 同目录 `run_worker.sh`，176–182 行 | Node 环境代理；NO_PROXY |
| 同目录 `molclaw.cordis.patch.yml` | MCP 四小时工具调用预算 |
| `slime-wd/deepseek-harness/packages/mcp/mcp-client/src/transport.ts` | StreamableHTTPClientTransport 使用默认 fetch，只设置认证 headers，没有定制 HTTP dispatcher |
| 同目录 `tools.ts`，callToolUncached/createExecutor | SDK request timeout 与 signal；收到 MCP isError=true 也会抛 Error。因此模型可见 Error 文案本身不能证明由哪一侧生成 |
| `slime-wd/deepseek-harness/packages/core/tools/src/index.ts:608`，errorMessage | 只保留 `.message`；cause/code 丢失 |
| 已安装 SDK `dist/esm/client/streamableHttp.js`，约 289–306 行 | POST JSON-RPC 使用 `(this._fetch ?? fetch)`；Accept 同时支持 JSON 与 event-stream |
| Node v24.19.0 内置 Undici 源码 | headersTimeout、bodyTimeout 默认 `3e5` 毫秒；MCP timeout 不会自动覆盖它们 |

SDK 实体目录：
`W/deepseek-harness/node_modules/.pnpm/@modelcontextprotocol+sdk@1.29.0_zod@4.4.3/node_modules/@modelcontextprotocol/sdk/`。

关于超时的精确含义：headersTimeout 是等待响应头；bodyTimeout 是响应体数据空闲间隔，**不是整个请求总时长**。SSE 已返回头并持续发数据/心跳时，不能按“工具超过五分钟必失败”推断。若 fetch 已返回响应头，后续流错误也可能表现为 terminated/SSE disconnected，而非裸 fetch failed。需记录失败发生在头部之前还是之后。

关于文案：Node fetch 网络错误可抛 `TypeError('fetch failed', {cause: ...})`；但服务端也可能在工具错误正文里返回类似文字，再被 bridge 抛出。只有 HTTP 侧诊断或原始响应可以区分来源。

## 5. 历史现象与样本位置

用户提供（原调查者未独立核验另一机器文件）：200 条完整原始 session，4278 次 MolClaw 调用，精确 fetch failed=0，受影响轨迹 0/200；另有其他工具失败/恢复，11 条待科学完整性复核。该科学复核不是此次 fetch 排障任务。

**跨语言/SDK的错误命名可能不同**：若 CPU 用 Python/httpx，连接异常可能叫 ReadTimeout、ConnectError、RemoteProtocolError 等；不能只按 JS 文案搜索便宣称两边网络错误率可比。先核对 CPU 客户端及所有真实工具错误类别。

历史 MS-3：

| 环境 | 出现过 fetch failed 的题数 | 出现过本地沙箱不可用 |
|---|---:|---:|
| 原版 L1，0909c | 19/25 | 24/25 |
| 原版层级，0909c | 22/25 | 23/25 |
| 旧 SFT L1，0909c | 14/25 | 16/25 |
| 原版 L1，后来串行 0910a，全量结束后 | 17/25 | 0/25 |

这些是“曾出现”的计数，不表示未恢复、不等于最终错误率。后来的串行结果证明并发不是已经成立的唯一根因，但仍无法排除负载影响。旧三组逐工具 fetch 次数总和为原版 L1 417、原版层级 540、旧 SFT 304；详见原统计，不能与表中“题数”相加。

低学习率 SFT 的 MS-2：22 次 fetch failed，涉及 13/37 题；Boltz2 21 次、sequence retrieval 1 次。旧 SFT MS-2 未记录 fetch failed，但有沙箱错误，因此分数差不能直接归因为学习率。

三个已逐行核对的 case（文件均为 `diagnostic_transcript.json`，相对 W）：

| 环境/题目 | 文件 | 调用行 | 错误返回行 | 工具 / 关联键 |
|---|---|---|---|---|
| 原版 L1 / MS3-007 | `outputs/dsh_molbench_evals/aligned-v4-9b-orig-l1-ms3-mo-full-0909c/results/molbench_ms3_007/diagnostic_transcript.json` | 3706–3716 | 3717–3745 | `retrieve_protein_structure_by_pdb_id`；pdb_id=5J2Z；seq=231→232；callId=`call_62e95e1a2d5a45a6acbba5f7` |
| 小学习率 SFT / MS2-001 | `outputs/dsh_molbench_evals/aligned-v4-9b-lr2e6-l1-all5-0910b/results/molbench_ms2_001/diagnostic_transcript.json` | 4124–4134 | 4135–4163 | `pred_binding_affinity_boltz2`；seq=747→748；callId=`call_ca9ef56bb52d47c083da1dfc` |
| 原版层级 / MS3-001 | `outputs/dsh_molbench_evals/aligned-v4-9b-orig-hier-ms3-mo-full-0909c/results/molbench_ms3_001/diagnostic_transcript.json` | 5150–5160 | 5161–5189 | `pred_binding_affinity_boltz2`；seq=3255→3256；callId=`call_cab1ba48ba714d5dabdd0457` |

工具公开名称实际带 `mcp__molclaw-scp__` 前缀。三个返回均为 text=`Error: fetch failed`、isError=true。Boltz 蛋白和分子参数很长，直接读上述调用行，不要从摘要重新拼写输入。

## 6. 已做实验：真实、失误与模拟分开

所有结果在 `P/reports/fetch_transport_diagnosis_20260911/`，文件内时间为 UTC；本地聊天/作业日志通常为 UTC+8。

| 文件 | 实验与解释 |
|---|---|
| `gpu_relay_retry.jsonl` | GPU 经正式隧道、relay、出口代理；tools/list=81；正确参数 is_valid_smiles 三次均成功，约 0.26–0.52 秒 |
| `login_upstream_correct_args.jsonl` | CPU 登录节点绕过 SSH、Python relay，直接经集群代理；相同正确轻量调用 3/3 成功。不是公网无代理直连，也不是用户另一台采集 CPU |
| `gpu_pdb_replay.jsonl` | GPU 重放 5J2Z 检索，约 4.7 秒成功；只证明当时可成功，不证明历史稳定 |
| `gpu_relay.jsonl` | 第一次诊断 SSH 握手被 reset；没有执行到 MCP 探测。不证明当时已有长隧道也断开 |
| `login_relay.jsonl`、`login_upstream_proxy.jsonl` | 初次探测误用 smiles 参数，服务端正常返回参数校验错误；可证明收到 HTTP 响应，不能计为科学工具成功 |
| `drop_connection_reproduction.mjs/.log` | **本地故障注入**：服务读请求后断开；实际 Node fetch 输出 fetch failed / UND_ERR_SOCKET；不是远端 MCP 故障 |
| `observer_check.log` | 手工发布诊断事件的检查；同样不是生产故障 |
| `probe.mjs` | 可复用隔离探测，见下面命令 |
| `REPORT.md`、`manifest.json` | 上一轮诊断结果及当时文件哈希 |

初次错误参数是 `{"smiles":"CCO"}`；正确参数为 `{"smiles_list":["CCO"]}`。原始失误日志故意保留，不要拿它当 MCP 计算失败证据。

探测中 GET 返回 405，而初始化和工具调用 POST 成功：这是 SDK 尝试可选 SSE GET 的响应；不能把该 405 单独算作工具失败。初始化后的 notification 返回 202 也不等于报错。

没有在本轮调用昂贵 Boltz/docking 来碰运气复现；没有迁移评测架构，没有重启 relay/工具服务。

## 7. 被动诊断器：已加什么、还缺什么

文件：`W/molclaw-mcp-relay/trace_fetch.mjs`。

新 worker 的 DSH 启动包含：

```bash
node --import "$worker_root/molclaw-mcp-relay/trace_fetch.mjs" \
     --import tsx/esm apps/cli/src/bin.ts web --no-open
```

订阅 `undici:request:create`、`undici:request:headers`、`undici:request:error`；仅匹配 `scp.intern-ai.org.cn`。输出在 `W/outputs/<INFRA_NAME>/dsh-start-*.log`，前缀 `[mcp-http]`，包含 UTC、method、进程内请求 ID、耗时、HTTP 状态、error code/cause。没有 headers、body、密钥，不改请求与模型观察。

局限必须知道：
- 只对加载它的**新 Node 进程**生效；19:21 已启动的首个补跑 worker 没有热加载它，后续 worker 才会加载。要看真实进程命令/日志，不要仅看磁盘脚本便假定正在生效。
- 当前 request_id 是**本地递增编号**，不是 DSH callId、JSON-RPC id 或服务端 request_id；日志重启后编号重置。
- 当前不记录 tool 名、调用参数、正常响应结束时刻；并发下只有时间与 method 不足以唯一映射工具。必要时在 MCP 边界补一条最小关联记录：DSH callId/JSON-RPC id/tool → HTTP 本地 ID/UTC；不把完整正文或密钥打进日志。
- 只观察特定域名的 HTTP 请求；代理 CONNECT/DNS 阶段是否也产生对应 request:error，要以实际故障验证。未见 trace 不代表没有网络故障。
- 当前 relay 日志主要是连接创建、字节量、EOF；并非每行都有时间。连接 ID 内的 epoch 是创建时间，**不能当每个事件的时间**。旧日志不足以逐秒还原断线。
- `bootstrap.log` 混合短 SSH 命令及长期转发；“Connection ... closed”本身不是断隧道证据。RSA hostkey signature 警告也没有证据与工具失败因果对应。

## 8. 接手后的最短调查路径

### 8.1 对齐另一台 CPU 的真实配置

先找到那 200 条的采集脚本、原始 session、run manifest，而非仅看清洗后数据。记录：MCP endpoint/协议、客户端语言/SDK/HTTP库版本、代理路径、连接复用、请求/读取/总超时、并发、工具分布、日期和用户报告的服务端修复时间/版本。认证只比较账户/路由差异，不展示 key。

用真实 tool result 与异常类别核对“零 fetch”；分开连接失败、HTTP 非成功、收到业务失败、模型复述。另 11 条科学完整性问题不并入传输失败统计。

### 8.2 无需模型的匹配对照

保持 endpoint、MCP SDK、工具参数、同一短时间窗口一致，顺序执行，避免新并发给服务器施压：

1. GPU 经现有 SSH+relay 路径。
2. 本项目 CPU 直接经集群 HTTP 代理（只移除 SSH+relay）。
3. 用户另一台 CPU 的原生产路径，并再用相同 Node/SDK 做一次匹配探测（区分语言/HTTP客户端差异）。

先 tools/list、正确 is_valid_smiles，再重放 5J2Z；不要一开始批量 Boltz。轻量全成功只能说明当前短连接可用，不能排除长请求超时。若失败集中慢工具，再选一个真实历史慢调用做有限、串行的对照，并采集响应头时刻与流心跳间隔；不要给正常题制造大负载。

必要时使用隔离测试服务构造延迟响应头、响应体空闲、连接中断，验证具体错误如何呈现。模拟只能证明机制，不可替代真实服务证据。一次只改变一个变量，例如仅绕过 SSH，或仅修改 HTTP timeout；不要同时换代理、SDK、服务端、并发后声称找到了某一根因。

### 8.3 捕获失败后决定查哪一层

| 观察 | 优先定位 | 不能直接推断 |
|---|---|---|
| UND_ERR_HEADERS_TIMEOUT，约 300 秒、无响应头 | 客户端 headers timeout，核对代理/网关是否已接收并持续计算 | 不能说远端工具未执行 |
| UND_ERR_BODY_TIMEOUT，已收到响应头后长时间无数据 | SSE/响应体空闲、心跳、代理缓冲/读取超时 | 不能按总执行时长判断 |
| UND_ERR_SOCKET / ECONNRESET / EPIPE | 同时段 SSH、relay、出口代理、网关连接日志 | 单凭 socket 错误不能认定哪一跳主动关闭 |
| ECONNREFUSED / CONNECT_TIMEOUT / ENOTFOUND / EAI_AGAIN | 代理监听、隧道是否存在、TCP/TLS/DNS 建连 | 不能立即归为工具算法失败 |
| HTTP 429/502/503/504 | 对应响应的网关/出口代理与 Retry-After（如有） | 与未收到响应的 fetch failed 分开 |
| HTTP 成功 + MCP isError/业务 status=error | 工具或下游业务日志 | 不能并为网络 fetch 错误 |

以 UTC 对齐请求开始、收到头部、失败与服务端处理时刻。若能访问网关，查是否收到 JSON-RPC 请求、是否交给工具、是否已算完但响应连接消失。已有 trace ID 不是服务端 request_id；必须建立映射。

### 8.4 修复边界与结束条件

有真实错误码/时序支持后，做相应最小修复：连接生命周期问题修转发/连接复用；超时不一致统一合理预算或保持流心跳；服务端/网关主动断开修对应环节。不要先统一把全部超时调成四小时，也不要用大量重试作为“已修复”的证据。

至少用原失败请求在相同条件下复测；分别说明“观察到什么”“改变什么”“哪条证据支持因果”“还有什么不能排除”。若未复现或无法访问网关，应明确停留在已验证的范围，不给虚假的唯一根因。

“CPU 上运行 DSH/MCP，GPU 只提供模型 API”可以作为隔离转发影响的后续架构对照；它会把跨机链路移到模型 API 一侧，不是让网络风险凭空消失。本次未实施，正式切换应单独记录实验环境版本。

## 9. 可直接使用的命令

### 只读查看作业与日志

```bash
P=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe
W="$P/slime-wd"
rjob list --namespace ailab-ma4agismall
rjob get JOB_NAME --namespace ailab-ma4agismall
# 从本次 rjob get 输出重新取 replica，不能复用下面的历史 pod 名。
tmux list-windows -a
rg -n '\[mcp-http\]' "$W/outputs/INFRA_NAME/dsh-start-0.log"
```

历史访问目标（仅供定位，必须重新查询）：`av4-failed-orig-l1-0911b-w1-hsb4v.sunxiangyu+root.ailab-ma4agismall.pod@h.pjlab.org.cn`。

### CPU 登录节点经出口代理探测

```bash
cd "$P"
DRUG_PROJECT="$P" SECRET_FILE=/home/sunxiangyu/.dsh/molclaw.env \
NODE_USE_ENV_PROXY=1 \
HTTP_PROXY=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128 \
HTTPS_PROXY=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128 \
"$W/outputs/dsh_eval_runtime/node-v24.19.0/bin/node" \
  reports/fetch_transport_diagnosis_20260911/probe.mjs
```

若要经本机 Python relay，把两个代理变量改为 `http://127.0.0.1:13208`。探测时检查继承的 NO_PROXY/小写变量，确保没有误绕路；不要在含凭据的环境直接 dump 全部 env。

### GPU 上经现有隧道探测

通过重新解析的 SSH target 登录，在 GPU worker 内执行：

```bash
P=/root/slime_sxy/group-space/sunxiangyu/drug-pipe
DRUG_PROJECT="$P" SECRET_FILE=/root/.dsh/molclaw.env \
NODE_USE_ENV_PROXY=1 \
HTTP_PROXY=http://127.0.0.1:13208 HTTPS_PROXY=http://127.0.0.1:13208 \
"$P/slime-wd/outputs/dsh_eval_runtime/node-v24.19.0/bin/node" \
  "$P/reports/fetch_transport_diagnosis_20260911/probe.mjs"
```

不设 PROBE_TOOL 时三次 is_valid_smiles。加 `PROBE_TOOL=retrieve_protein_structure_by_pdb_id` 会只查一次固定 5J2Z。**该脚本只为这两个探测写了参数，不能随意更换工具名而不改参数。** 脚本总预算 90 秒、单次工具请求 30 秒；它不是正式四小时测评，也不能用它证明长请求稳定。硬预算到期也不代表服务端一定取消了工作。

### 检查实际 Node 内置版本及超时

```bash
"$W/outputs/dsh_eval_runtime/node-v24.19.0/bin/node" - <<'JS'
fetch;
console.log({node:process.version, undici:process.versions.undici});
const lines=process.binding('natives')['internal/deps/undici/undici'].split('\n');
for(let i=0;i<lines.length;i++)
  if(/this\[k(?:Body|Headers)Timeout\] =/.test(lines[i])) console.log(i+1,lines[i]);
JS
```

这是对当前二进制的内部源码读取；换 Node 版本后内部实现可能变化。官方 fetch/dispatcher 说明：[Node 文档](https://nodejs.org/api/globals.html#fetch)。

## 10. 当前评测与 git 状态，不要误操作

最后一次查看队列状态文件显示第 1/4 项（状态字段 running_or_queued 不等于调度器实时状态）；实际上一轮 rjob get 为 Running。当前以**答错题**而非 fetch 出现频率筛选：

| 模型 | 待补跑题数 |
|---|---:|
| 原版 9B / L1 | 78 |
| 原版 9B / 层级技能 | 75 |
| 旧 SFT / L1 | 88 |
| 小学习率 SFT / L1 | 100 |
| 总计 | 341 |

队列目录 `E/failed_questions_0911b/`；入口 queue.py；manifest.json、selection_audit.json、各模型 *-ids.json；tmux `failed-questions-0911b`。首个 job `av4-failed-orig-l1-0911b-w1`；结果 `W/outputs/dsh_molbench_evals/aligned-v4-9b-orig-l1-failed-questions-0911b/`；基础设施目录 `W/outputs/infra-aligned-v4-9b-orig-l1-failed-questions-0911b-w1/`。确认新 worker 是否加载 trace 要看实际命令，不要只看队列名。

作业按模型合并该模型的失败题，模型/题目/工具调用串行、两卡；每题最多额外两次基础设施重试，worker 最多额外两次恢复。相同工具和参数后续成功视为已恢复；科学错/格式错不触发自动 retry；耗尽保留最后答案，不取最高分。详情在现有 runner，不另造重试框架。

341 题按容忍代码块的科学 scorer 结果筛选，历史任一已完成回答通过即排除；MS3 看 Top-3 命中，mol-opt 用 improvement>0。**这批是诊断补跑，不是独立全量 benchmark，不能把择题重试后最佳结果冒充正式准确率。** MS3 不加回 60 项完整性、重复或候选外字符串门槛。此前 621 题/17 setting 队列已撤销，不要重新启动。

主仓库已推送 commit `54bfd3165b112bbfb37808e1f90fcbf2c0817c1a`（main）；此后新增诊断 trace、run_worker 加载改动和本交接材料尚未另行 commit/push。**仅 clone 该 commit 不会包含最新 trace；需要共享文件或本次交接副本。** deepseek-harness 是独立子模块，存在本地文件读取相关改动；不要 reset/覆盖或假定它随主仓库 push 自动提交。

## 11. 证据总索引与旧材料纠错

- 本文：`P/docs/HANDOFF_FETCH_FAILED_ROOT_CAUSE.md`，当前接手主入口。
- 新网络诊断：`P/reports/fetch_transport_diagnosis_20260911/`，完整实验日志、probe、本地模拟和当时哈希。
- 旧综合报告：`P/reports/evaluation_incidents_20260911/REPORT.md`；同目录 fetch_failed_by_tool.json、structured_tool_errors.json。后者含 232 条结构化工具业务失败，**不是 232 条 fetch failed**。
- 原服务端交付包：`/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug_wd/mcp_server_diagnostics_20260911_153319.tar.gz`，以及同名解压目录、.sha256。含旧报告、轨迹与服务端业务失败证据；适合查原始资料，但不是当前根因结论。
- 旧整组筛选审计：`E/infra_rerun_0911a/prior_results_audit.json`，包含逐模型/任务历史频次。旧整组重跑方案已取消。
- 当前诊断补跑：`E/failed_questions_0911b/manifest.json` 和 selection_audit.json。

需要主动纠正的旧说法：
1. rjob 成功或 completed 不等于工具健康；旧 infra_failure_count 漏掉完成轨迹内的工具故障。
2. “三个模型并行造成过载”只是曾经的猜测；后续串行仍发生 fetch failed。
3. “服务端修复已完成”来自用户描述，原调查者没有对应 server commit/变更清单/部署时间，不能据此认定历史或当前链路所有层已修复。
4. QuickVina 断言失败、PDBQT 解析失败、Boltz 缺产物是真实业务失败证据，不是裸 fetch failed 的根因证明。不要把这些远端 request_id 强行对应上表三个 fetch case。
5. 本地 bash 沙箱不可用是另一问题，已补运行产物并做过 worker 验证；它不经过 MCP 网络。
6. 工具错误正文也可能含 fetch failed；必须用 HTTP 事件与原始 MCP 响应判断生成位置。

接手后的理想交付：一条真实失败的两端证据链、一个被验证的机制、对应的最小改动及匹配复测结果。若仍未能确认，则给出已经排除的范围和所缺的具体一条证据，不凭经验写“根因已修复”。
