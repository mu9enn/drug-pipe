# MS-3 / MO-Opt / MO-Edit：9B 三组测评

本轮只运行原版 Qwen3.5-9B/L1、原版 Qwen3.5-9B/层级技能、596 条数据训练的 SFT/L1。没有 SFT/层级技能，没有新增训练。入口为 `slime-wd/dsh-molbench/pretrained_matrix/submit_extended_9b.py`。

## 固定输入与服务

- MolClaw commit `180abb7679ffc5b1ca9974e7390e141d8098f642`：MS-3 25 题、MO-Opt 39 题（LogP 11、QED 20、Solubility 8）、MO-Edit 39 题（add 10、delete 9、sub 20）。每组 103 题。MO 不包含 target optimization。
- MO 六个源文件已与该 commit 的 GitHub 原始文件逐字节核对。原文件保存在 `slime-wd/molbench/upstream/<commit>/data/molbench-mo`，发布题单单独存放在 `aligned-mo`，不改变已有 MS 发布或历史结果。
- MO-Opt 原示例包含尾随逗号。本次仅替换输出说明，保留原字段名：MO-Opt 为 `{"Final Target Molecule":"<modified molecule SMILES>"}`；MO-Edit 为 `{"output":"<modified molecule SMILES>"}`。不额外要求 evidence，不限制新分子属于输入候选。合法 SMILES 与科学正确性由 scorer 判断。
- MS-3 使用现有 `ranked_smiles` + `evidence` 契约，要求完整候选排列，并保留题目排序方向。
- 三组同题、同 system、同 88 个工具 schema、同 tokenizer/chat template、同 native thinking、greedy、262144 context、16384 单次响应、每题 14400 秒、并发 2、仅基础设施故障最多重试 2 次。
- 三组统一 2×H200、TP2、seed 42，复用同一 SGLang/DSH launcher。记录实际完整 ServerArgs，启动时检查实际 TP 与 seed。greedy + 固定 seed 不宣称位级确定性；外部科学工具的动态结果也不宣称冻结。
- L1 为相同 52 个技能；层级为 68 个，增加 L2/L3 方法论信息。原版与 SFT 权重来源、实际服务版本及模型文件哈希由运行记录保存。

## 测试隔离与解释边界

对实际 596 条训练发布复查，112 道 MS 的任务组重叠为零。MO 按训练题目里的 canonical isomeric 分子匹配后人工核对目标：

- MO-Opt/QED source ID `48b3a4aa-d573-44f9-8fda-dfc4757a3aaa` 与 3 条 E2E 训练轨迹同起始分子、同 QED 优化目标。完整 39 题结果保留，但不能称为全部未见任务；三组同时报告剔除此题后的 38 题结果。
- MO-Opt/QED source ID `eee48d84-9dd3-4bef-aa00-89556690d5d3` 与 1 条训练 VS 任务共享候选分子，目标不同。记录分子重叠，不当作相同优化任务。
- 未发现 MO-Edit 起始分子匹配。上述检查不等于 scaffold-disjoint 划分。

审计：`drug_wd/drug_pipe_regular_v1_20260908/experiments/ms3_mo_0909c/holdout_audit.json`。不因本次审计修改已训练 checkpoint，不根据测评分数选择 checkpoint。

## 评分与执行

严格 JSON、容忍唯一 JSON 代码块、容忍唯一完整 JSON 对象三种口径在运行前声明，对三组完全一致。后两种只在保存的最终回答上离线处理，不读取 reasoning、工具观察或标准答案来恢复输出，不回写原始回答。所有选定题保留在对应分母；缺失、模型失败和不合法输出不被过滤。

使用冻结 MolClaw scorer；MO 调用现有 ChemCoTBench checkout `19cef8c900db689cacb0cfaaf4a452152ae709d8`，逐源文件哈希写入运行 manifest。登录侧独立评分 venv 使用 `mo-score-requirements.txt` 中的版本，避免改变推理环境。

MS-3 报告上游 Top-3 命中及 GT Top-3 平均排名；MO-Edit 报告 correct rate；MO-Opt 分属性报告 improvement、success rate、scaffold 等上游指标，不将三种任务都叫作 accuracy。另保留格式合规与故障统计。

每个 GPU rjob 的前台 entrypoint 是实际 worker：模型加载与工具调用探针 → harness 首轮协议检查 → 全部选定任务 → 协议审计及基础设施重试 → 退出释放 GPU。登录侧 tmux 负责密钥安装、MCP 隧道及 GPU 结束后的离线评分，没有 GPU keepalive。源数据发布、固定预测 scorer 回归和 shell 检查通过后才提交。

结果位于 `slime-wd/outputs/dsh_molbench_evals/aligned-v4-9b-{orig-l1,orig-hier,sft-l1}-ms3-mo-full-0909c`；每组 `format_sensitivity/comparison.json` 包含三种解析口径及 MO-Opt 的 38 题补充结果。

2026-09-09：按用户要求，三个未获得资源的 0909b 四卡任务已取消；以 0909c 新运行目录重新提交两卡 TP2 任务，保留原 manifest 作为取消记录。
