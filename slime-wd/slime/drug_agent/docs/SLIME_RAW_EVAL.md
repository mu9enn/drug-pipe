# slime-raw 严格工具评测骨架

`slime-raw` 只实现以下闭环：

1. 新任务 prompt 加上模型无关的严格协议、实时工具 schema 和任务最终输出 schema。
2. LLM 生成一个完整决策：`tool_call` 或 `final_answer`。
3. 严格解析该决策；任何协议外文本或非法 JSON 立即判为格式失败。
4. Python 校验并执行合法工具调用，将真实结果序列化为 `observation`。
5. 原样追加模型输出和 observation，继续下一轮，直到合法 final answer 或失败边界。

## 明确排除的行为

- 不添加 `/no_think` 或任何模型、checkpoint 专属提示。
- 不挂载、提示或强制读取 `SKILL.md`；`Read/Grep/Glob` 只访问 task workspace。
- 不恢复多余闭合标签、逗号分隔 JSON、Markdown fence 或模型 transport token。
- 不为 `tool_call`/`final_answer` 添加特殊 generation stop。
- 不删除、重写或清理模型输出 token。
- 不压缩、摘要或裁剪历史；完整历史超过 prompt budget 时直接失败。
- final answer 的 task type/schema 不正确时直接失败，不注入纠错 observation。
- 工具名必须与公布 catalog 中的规范名称完全一致，不接受 MCP 前缀别名。

工具执行失败和 JSON Schema 参数错误仍会作为真实工具错误 observation 返回；这是工具环境本身的语义，不是模型兼容恢复。

## 启动

使用与现有 Slime MolBench 入口相同的 checkpoint、模型拓扑和数据参数，只把入口换成：

```bash
MODEL_CHECKPOINT=/path/to/toolrl \
MOLBENCH_SUITES=molbench_ms1,molbench_ms2 \
bash drug_agent/scripts/run_slime_raw_eval.sh
```

落盘的 `run_manifest.json` 会记录：

- `settings.eval_profile = "slime-raw"`
- `settings.protocol = "canonical_react_xml_exact"`
- `settings.model_specific_compatibility = false`
- `settings.context_policy = "full_history_no_compaction"`

raw manifest 中的 L1 skill root/hash 为 `null`，tool catalog 也不会公布 skill 路径。

每条 trace 同样记录 `rollout_mode = "slime-raw"`，便于与 compat、DSH 或普通 Slime 结果严格区分。
