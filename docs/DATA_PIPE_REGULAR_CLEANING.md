# v8 轨迹要求进入常规 data-pipe

2026-09-08。本次只整顿 CPU 数据流程，不提交 rjob，不恢复训练、评测或自动队列。原 596 条发布及历史实验保持原样。

处理原则：能用于训练的轨迹尽量保留。Python 负责确定性转换和少量硬约束；LLM 负责科学语义与措辞。重复措辞、计划不够漂亮、推理不够丰富不作为丢弃理由。格式合规不等于科学正确。

## 要求、原有缺口及现在的位置

| 要求 | 修改前的落地情况 | 当前主线位置与处理 |
|---|---|---|
| AC/PF/VS/KG/E2E 使用统一题目和最终 JSON 契约 | 生成、采集和导出已有共享契约 | 保持 `output_contracts.py`；采集只验证已发布题面，历史题面迁移仍须显式执行 |
| 全部 112 道 MS 题保留；反向 AC 也隔离 | 分组和科学题面匹配已落地；来源 ID 只在历史发布入口使用 | 采集/生成继续分组隔离；Python 和 SFT 导出也接收已有 `source_task_ids`，不把普通行号猜成基准 ID |
| 解题工作区不提供标准答案 | 私有 `question.json` 在执行目录的祖先目录 | raw 阶段使用独立执行目录，工作区 `question.json` 只有公开题面、候选和任务类型；私有标签留在归档供离线评分 |
| Claude、DSH 两条采集入口均可使用 | 两条入口存在，默认技能场景和本地工具暴露不一致 | 默认 L1 工作区与统一题面前缀；Claude 限定七类本地工具，DSH 科学采集禁用额外工具插件；MCP 保持显式配置 |
| native reasoning、并行调用、observation 绑定、路径语义 | semantic builder 和 SFT adapter 已实现 | 继续复用，不另建轨迹格式；不把观察重排成与调用不匹配的顺序 |
| 科学结果表不能只保留头尾 | 长结构化 observation 有通用裁剪路径 | Python 保留结构化表格完整内容；仍去除二进制大块，完整轨迹是否超长由最终 tokenizer 判断 |
| L1 catalog、正文、资源路径和首用加载对齐 | 有适配，但在 LLM 清洗之后；错误 skill 可被投影成成功 | 移到 LLM 之前；成功加载投影为当前 L1，失败调用及错误 observation 原样保留，不把失败记作已加载 |
| 历史层级技能别名与叙述清理 | 已有全字段替换，但会生成 “legacy note” 处理话术 | Python 做确定性别名转换，取消新增处理话术；LLM 清理叙述并保留失败后的真实重试 |
| 候选外答案、等价 SMILES、缺项排序可恢复 | 严格 Python 校验提前拒收；84 条恢复依赖历史专用处理 | `answer_recovery.py` 接入 Python 与 LLM；明确选择、唯一保留立体化学的候选映射、已观察数值排序优先确定性恢复，其余交给 LLM |
| 未确定的 VS 尾部可以保留 | 历史处理已有，常规流程没有 | 必须先有实际排序证据；未知候选放末尾并说明优先级未确定，字典序只用于打破并列，不冒充亲和力测量 |
| 无科学结论的中断样本丢弃 | 两条历史样本已剔除 | 没有最终回答的结构不导出；无任何排序/选择依据的不凭空补答案，留在 pending/隔离侧文件 |
| 禁止从标准答案补标签 | 历史 PF 已发现，常规流程缺少针对性检查 | Python 和 SFT 拦截明确答案文件查阅/引用；普通“实验结果是 ground truth”措辞不等同于泄漏 |
| 恢复等级和处理说明不进入训练 | 历史 appendix、evidence 中有处理计数 | 等级、事件引用、修复原因只进侧文件；发布/导出去掉已知处理字段，历史 appendix 按侧文件精确移除，不正则重写科学正文 |
| LLM 只能改 reasoning 和被授权恢复的终答 | reasoning patch 已有限制，缓存绑定不足 | 保留工具历史不可变检查；增加可选终答 patch 与现有事件引用；缓存绑定输入、提示、模型和 skill；提示明确禁止把后续观察搬进前面决策 |
| LLM 失败不能大批丢弃本来有效的轨迹 | 调用或措辞校验失败会进 pending | 终答及结构有效则保留并记 warning；终答仍无依据/不合规才 pending；HTTP 500 重试也计入次数上限 |
| SFT 不能绕过候选、测试隔离和工具绑定检查 | 已有独立检查 | 继续复用共享契约和绑定检查；真实失败调用允许保留错误参数，不要求其伪装为成功调用 |
| 正式发布必须通过实际 tokenizer、mask、长度门控 | `gate_release.py` 已存在，常规 shell 没有调用 | `run_cleaning.sh` 最后调用 `publish_dataset.py`，复用原 gate；固定完整工具/系统/catalog，整条保留或超长隔离，禁止截断 |
| 发布可追溯、不覆盖母数据 | 历史发布有 manifest，常规输出缺最后一步 | 新目录发布 semantic、SFT、training、审计、文件哈希与 token gate manifest；目录存在则不覆盖 |

## 三个处理时机

1. **raw 采集**：公开题目契约、测试隔离、L1 场景、限定工具、私有标签与执行目录分离；完整保存会话和调用绑定。此处不为格式错误重跑挑答案。
2. **Python**：结构构建、路径和 skill 适配、明确证据恢复；LLM 后只验证终答契约及既有结构约束。科学结果表保留，措辞问题不做硬门槛。
3. **LLM**：去除教师环境话术、改善推理表达、恢复仍待处理的终答；保持每步只使用此前信息，保留科学不确定性。所有处理等级写侧文件。有效样本遇润色失败仍可导出。

最后用实际 checkpoint tokenizer 执行一次全量门控。它是训练兼容性检查，不再添加另一套科学正确性评分器。

## 常规入口

从项目根目录运行；`RELEASE_ROOT` 必须是新目录，tokenizer 必须是已有本地 checkpoint。

```bash
bash data-pipe/scripts/run_cleaning.sh \
  --results-root "$RAW_RESULTS" \
  --deployment-tool-set data-pipe/configs/dsh_molclaw_tool_set.json \
  --tokenizer slime-wd/data/Qwen3.5-9B \
  --release-root "$RELEASE_ROOT" \
  --harness deepseek
```

`--harness claude` 同样支持。LLM 清洗会调用所选服务；本次只做 mock 验证，没有启动在线采集/清洗。历史 semantic 母数据可直接进入 `pipeline.cleaning.llm_clean --input ...`，但其题面仍须先通过显式历史迁移。不要把新采集入口变成静默改题入口。

已有审阅记录的历史发布可用以下入口去除处理说明并重新门控；`--reviews` 仅提供要移除的精确 appendix，不从该文件重新补标签：

```bash
PYTHONPATH=data-pipe python -m pipeline.cleaning.publish_dataset \
  --input "$OLD_RELEASE/semantic_trajectories.jsonl" \
  --reviews "$OLD_RELEASE/reviewed_reconstructions.jsonl" \
  --audit "$OLD_RELEASE/answer_audit.jsonl" \
  --audit "$OLD_RELEASE/quarantined.jsonl" \
  --output-root "$NEW_RELEASE" --tokenizer slime-wd/data/Qwen3.5-9B
```

## 验证范围和限制

针对这次行为变化验证候选外答案、部分排序和重复、未确定尾部、空筛选、明确泄漏与普通科学措辞、真实 skill 错误、LLM 失败保留、Claude/DSH 模拟执行及 SFT 绑定。不进行在线科学评测或逐样本额外 LLM 审判。

工作区分离是避免意外读到标签，并非操作系统权限沙箱。LLM 的科学判断与逐步因果性受提示约束，Python 不假装能证明其科学正确性。Claude 与 DSH 原始消息形态不同，通过 SFT adapter 对齐；这次不声称新增的 raw 配置已经完成真实服务首轮请求逐字回放。

新发布目录：`/mnt/shared-storage-user/sdpdev-fs/sunxiangyu/drug_wd/drug_pipe_regular_v1_20260908`。它是数据流程验证发布，不自动替换暂停中的训练/评测实验 manifest，也不沿用旧实验得分作为其结果。

验证结果：58 个相关回归测试通过，shell 语法和本次 diff 空白检查通过。实际 tokenizer 门控为 599 → 596；被排除的仍是原来的 3 条超长 KG。599 条的答案选择/排序/result 均未改变，84 份已知处理 appendix 在新训练文本中均无残留。训练文件 SHA-256：`1f0ae23b80d3d59288fb84f4d7723211fdd19fabddf5564048d12880bb34a07b`。完整长度、mask 角色检查及哈希以新目录的 `training/context_gate_manifest.json` 和 `release_manifest.json` 为准。
