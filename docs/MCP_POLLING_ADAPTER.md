# slime-sxy drug-pipe MCP 续查适配接入（2026-09-16）

范围仅 slime-sxy 的 drug-pipe。没有修改采集机或 MolClaw 服务端，没有重启现有测评/训练作业。

| 入口 | 修改前 | 修改后 |
|---|---|---|
| Data-Pipe run_kg_pipeline.sh -> launch_claude.sh -> run_claude.py（Claude/DSH API harness） | 临时配置是 SCP HTTP 直连 | 默认生成 stdio 适配配置 |
| run_claude.py 显式传入旧 SCP HTTP 配置 | 沿用直连 | 自动转为私有临时 stdio 配置，保留其他 MCP server，原文件不变 |
| slime 原生 MCPClient / run_molbench_eval.sh | Python Streamable HTTP 直连 | Python stdio ClientSession -> 同一个适配器 |
| 通用 DSH pretrained_matrix 默认模板 | HTTP 直连 | stdio -> 同一个适配器 |
| 当前 fresh_four_0916d CPU 评测 | 已接入静默续查适配器 | 保持原版本和运行配置不变，冻结源码哈希验证通过 |

共享入口是 `runtime/molclaw_mcp.sh`；配置生成器是 `runtime/molclaw_mcp_config.py`；可配置入口是 `slime-wd/molclaw-mcp-relay/polling-adapter/stdio_configurable.mjs`。它复用现有 adapter_server、polling_client_core、transport_recovery，不复制续查业务逻辑。原 stdio_adapter.mjs / run_mcp.sh 保留，避免改变已冻结测评。

原理：模型侧只有一个 tools/call；适配器对 SCP 使用同一个 jobId start/poll，每次等待最多 300 秒，后台科学工具继续运行。中间 running envelope 只被适配器消费，不作为 tool_result 返回，不发送 progress/logging 通知。最终科学结果或明确错误才送给 harness。保留限定的工具级传输恢复，新增入口初始化遇已知瞬时 fetch 错误时最多额外重试三次（1/3/9 秒）；不新增整题恢复。

网络仍是 CPU -> 实验室批准的 HTTP 代理 -> SCP。共享入口尊重已有 HTTP_PROXY/HTTPS_PROXY，因此旧 GPU 环境仍需要已有的批准网络出口；适配器不创造网络连通性，不是内网穿透。当前 CPU 评测无需 molclaw-relay。

凭据：支持 Data-Pipe 的 MOLCLAW_SCP_MCP_AUTH 和 slime 的 MOLCLAW_SCP_API_KEY，也支持 SECRET_FILE。普通 Data-Pipe 配置不写密钥，密钥由父进程环境继承；从用户原有的含凭据 HTTP 配置转换时，在权限 0600 的临时文件中保留必要凭据，进程正常退出自动删除。没有把密钥写入代码/报告或复制到外部。

Node 使用项目已有 v24.19.0，可通过 MOLCLAW_NODE_BIN 指定；SDK 使用已有 DSH MCP SDK 1.29.0。Python MCP v1 的 timedelta 和 v2 的 float timeout 均有兼容测试。外层工具预算为四小时，续查并不改变整题预算。

验证：

- 实际 launch_claude.sh 的 Claude 与 deepseek（DSH）两条路径使用假 runner 截获配置，确认 stdio、原 server name、四小时预算、strict 配置、凭据不在普通配置内容里。
- session_capture 的 DSH 配置转换保留 stdio；旧 HTTP 配置转换保留其他 server、原文件不改，临时文件权限 0600。
- 标准 MCP 客户端协议测试：内部两次连接错误、多次 running；带/不带 progress 支持均只收到一个最终结果、零 notifications、同 jobId 恢复。
- 原生 Python 客户端真实连 SCP，发现 81 个工具，is_valid_smiles(CCO) 成功，返回真实结果，无内部 running envelope。耗时约 28.25 秒。
- 原生客户端 mock 生命周期/超时类型测试通过；原有 online_tool_environment 17 项通过（其超时清理用例仍输出警告）。
- Data-Pipe 22 项回归：21 项首次通过；一项已有 10 秒上限的并行测试受冷启动影响超时，单独复测 5.79 秒通过。未放宽测试上限。
- Bash 语法及冻结测评源码哈希检查通过。

本次没有重新执行 >600 秒的真实科学工具实验。超过 600 秒依靠同一份已验收的后台续查实现；这次核对的是新增入口确实使用它、标准客户端不会收到中间状态。

检查时无活动 run_claude.py / launch_claude.sh / run_kg_pipeline.sh / run_molbench_eval.sh 进程，不存在需要中断切换的旧采集进程。已经手工启动且自行使用其他 MCP 配置的外部客户端不会被脚本热修改。

代码备份、部署映射、测试脚本与 live_native_result.json、live_native_adapter.jsonl 位于本目录。新代码尚未提交 Git，供后续统一提交。
