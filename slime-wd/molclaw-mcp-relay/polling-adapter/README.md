# SCP 等待消息适配：部署与验证

状态：10 秒与正式 300 秒间隔的真实 SCP 测试均通过，服务端及 CPU 上的标准 MCP 适配入口已部署。已为 0914e 基础设施故障诊断补跑配置静默适配服务；旧原始结果保留。

## 调用语义

闭源 harness → 标准 MCP stdio 适配服务 → 原 SCP HTTPS 地址 → 工具外层续查中间件 → 原科学工具。

harness 发起的外层 tools/call 一直等待，适配服务只在原工具完成后返回最终 CallToolResult。根据用户后续要求，等待期间不向 harness 发送任何 progress、logging 或心跳通知，内部状态只由适配服务消费；仅完成后返回最终工具结果。不要求修改 harness 内部 MCP 实现。

适配服务内部会发起多次短一些的 SCP RPC。每次内部 RPC 可以返回普通 JSON 等待状态，这些状态被适配服务消费，不直接暴露成外层调用结果。这不是让同一条 SCP HTTP 连接无限保持，而是避免一条 SCP 请求持续超过它的约 600 秒限制。

必须修改一次 MCP 配置，指向适配服务。原 SCP 地址本身没有被替换或修复；直接连接它且不启用续查的客户端保持原行为，也不会收到新增的等待状态。

## 已部署文件

服务端 `/data/lwj/wll/code/DrugAgentTools/sxy_sum/scripts/`：
- `molclaw_progress_wrapper.py`：外层注册中间件，普通心跳默认 300 秒。
- `polling_jobs.py`、`polling_worker.py`：只有显式 `_molclaw_poll` 请求启用，原参数传入原工具，独立工作进程执行一次。
- SQLite 结果位于 `runtime_logs/polling_jobs/`，权限限制在服务账户。相同任务 ID 和参数不重复启动；参数不匹配、未知任务续查会报错。

slime-sxy `/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/molclaw-mcp-relay/polling-adapter/`：标准 MCP 适配服务、`run_mcp.sh`、`mcp-config.json`。配置文件不包含密钥；启动器从该机已有凭据文件读取，使用实验室已有 HTTP 代理。

在 slime-sxy 上，把客户端的 molclaw-scp 配置替换为 `mcp-config.json` 中的条目即可。该示例配置在 CPU 登录节点执行。0914e GPU worker 使用独立的 molclaw.polling.cordis.patch.yml，将路径映射到 /root/slime_sxy，并在 worker 内启动同一轻量 stdio 适配程序，出网沿用已批准的 localhost:13208 relay 与实验室代理通道。

适配器等待预算为 4 小时；内部请求使用已有仅针对 SCP 的 4 小时 HTTP headers timeout 修复。客户端自身的工具调用预算仍需覆盖真实工具时长。正式等待间隔默认 300 秒，诊断用环境变量 `MOLCLAW_POLL_INTERVAL_SECONDS=10`。

## 当前证据

- `server_contract_test.json` 及结构化输出修正版测试：重复开始只执行一次、缓存最终结果不变、取消首个请求不终止启动、工具错误保留、普通调用不变。
- `test_standard_client.mjs`：标准 MCP 客户端有/无 progress 处理器时，等待状态都不产生外层最终响应；最后只收到一个最终响应。
- 真实 SCP `scp10_structured_client.jsonl`：99.165 秒完成，9 条等待通知，1 次成功最终结果。
- 真实 SCP `scp300_long_client.jsonl`：668.105 秒成功完成；300.283 秒、600.610 秒收到等待通知；仅一次最终结果，没有 fetch failed。
- 两次真实工具测试的最终 content 和 structuredContent 与服务端缓存完全一致。传输后省略了可选的 `isError:false`，语义相同；不是逐字节一致。见 `evidence/completion_audit.json`。
- 持久目录中的 MCP 配置通过真实轻量工具验证，2.504 秒成功。
- 首次真实测试暴露结构化输出缺失，已补齐。此前任务未重启，通过同一任务 ID 取回成功结果。
- 81 个现有输出 schema 均为允许任意属性的 object，可接受内部状态对象，工具 schema 无修改。
- 科学工具源文件修改前后 SHA256 均为 `956892b1828b50eb8bb40682848e576bb81810bc565eae50e2f3638ac36e6b13`。

静默模式追加验证：标准客户端有/无 progressToken 均收到 0 条通知、1 个最终结果；真实 SCP 测试 97.411 秒成功，内部 10 秒续查而外层通知数为 0。DSH 原生 stdio transport（含环境清理）也通过真实轻量调用，81 个工具可见。未进行 Claude Code 实测，不能声称已通过。

## 限制与回退

这针对长期无最终响应触发的 SCP 超时，不代表代理 CONNECT 502、工具内部错误等都已消除。工作进程消失会报告执行状态不确定，不自动重跑科学工具。状态与结果暂不自动清理，避免丢失可续查的任务；须在确认不再续查后按运维策略清理。

停止使用适配配置即可恢复原 SCP 直连行为，服务端显式启用逻辑不会影响未启用调用。若需恢复服务端旧包装器，必须等待活动工具结束后使用备份文件，不能直接杀掉正在执行的工具。科学工具源文件无须回退。


## 2026-09-14 CONNECT／连接中断恢复

客户端适配服务对 tools/call 内部 RPC 的已识别传输故障做至多 3 次额外重试，间隔 1、3、9 秒，沿用原 RPC 的超时截止时间及相同任务 ID／start 或 poll 动作。支持底层 CONNECT 502/503/504 与指定连接／socket 错误；不重试笼统的 fetch failed、认证／证书错误、原工具错误或用户取消。不改变科学工具计算逻辑；服务端既有任务 ID 去重保障重复请求只取原任务状态／结果。

不发送 progress、logging、心跳或内部重试通知给 harness。嵌套错误链与任务 ID 写入本机 `slime-wd/outputs/molclaw_adapter_logs/` 的每进程 JSONL，目录 0700、文件 0600；凭据值会脱敏。可用 `MOLCLAW_ADAPTER_LOG_FILE` 指定已有父目录中的文件。初始化故障也写本地日志，初始化重连仍由 harness 既有策略管理。

这能恢复短暂故障，不是修复出口代理本身；持续故障超过有限重试仍明确失败。上线验证与备份见项目 `reports/adapter_recovery_rollout_20260914/`。
