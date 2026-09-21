# AGENTS.md — Drug-Pipe

MolClaw 任务构造 → Claude 数据采集 → Qwen3.5 structured SFT 主线。本文件是**地图**：只放边界、授权分级和指针，不重复别处已有的规则。

## 指令优先级

1. 用户在本轮对话里的明确指令
2. 本文件
3. 项目 Skills（地图见 `.codex/skills/README.md`）
4. `docs/` 与代码注释

下层与上层冲突时以上层为准，并**指名文件与章节报告冲突**，不要静默取舍。

## 状态与冲突

- **易变状态只有一个家：`docs/STATUS.md`。** "当前是否暂停 / 当前发布哪一版 / 当前跑多大模型"一律以它为准；
  该字段为 `TODO` 时视为未知，**去问用户，不要从其他文档或 git 历史推断**。
- 本项目文档跨度大、日期多，已知存在互相矛盾的陈述。遇到不一致时指出文件与章节、说明你采用了哪一条；
  如果冲突影响**是否执行**（例如"训练当前是否被授权"），停下来问。

## 按任务读取

不要默认全读。按下表按需读取；表里没有的，先在 `docs/` 里搜，再问。

| 任务 | 读 |
|---|---|
| 当前是否允许提交 rjob、当前发布版本与模型规模 | `docs/STATUS.md` |
| 数据格式、字段契约、清洗顺序 | `docs/DATA_FORMATS.md` |
| 数据集与评测协议、held-out 隔离 | `docs/EXPERIMENT_ALIGNMENT.md` |
| 具体怎么跑（Tool-KG / Data-Pipe / SFT / 评测） | `docs/RUN_COMMANDS.md` |
| 已知坑与"本轮未验证"清单 | `docs/KNOWN_ISSUES.md` |
| 常规样本处理要求与阶段分工 | `docs/DATA_PIPE_REGULAR_CLEANING.md` |
| 数据流、"哪个事实由谁定义" | `docs/MAINLINE.md` 的 Authorities 表 |
| 训练配置、并行、显存、失败诊断 | Skill `slime-h-cluster-training` |
| 提交 GPU 任务、配额池 L1/L2、占卡生命周期 | Skill `h-rjob-submit` |
| 轨迹数据生产 / 清洗 / 审计 | Skill `manage-drug-pipe-trajectories` |
| 某个设计**为什么**这样、放弃了什么 | `docs/decisions/` |

## 项目不变量

- 正式离线 SFT **不执行真实工具**；真实 MCP 只在数据采集与显式 online 评测/调试路径。
- 旧 XML ToolRL / online runtime 已隔离到后续专项，SFT 入口与文档命令不调用它们。
- `raw` session 与 canonical 母数据**不可覆盖、不可删除**；修复新建 attempt 或校验后复用。
- Benchmark label 与 evaluator 指标**永不进入训练 prompt**。
- 训练长度限制用原生 template/tokenizer 审计决定，**不得靠截断或删除教师动作实现**。
- 结构清洗与学术正确性是两层独立验收；`accepted_count == N` 不等于科学完成。

## 授权分级

自行执行：读任何文件；在 `docs/`、`reports/`、`.codex/skills/` 下写文档；改本仓库代码并跑本地/CPU 测试；
只读状态检查（`nvidia-smi`、`rjob list`、`brainctl get`、进程与日志、`git status`/`git log`）。

必须先问：提交或删除 `rjob`/`rlaunch`；干扰运行中的实验、杀进程、重启 Ray；切换 cc-switch provider
或改 provider 配置与凭据；发布数据、覆盖 canonical 数据、删除 raw session；`git push` 与任何对外动作。

## GPU 与集群

- 提交走 Skill `h-rjob-submit`：**先选配额池**（L1 `ailab-agenttool` / L2 `ma4agismall`），再渲染命令。
- **绝不用 `sleep`、交互 shell、`tail -f`、detach 的 tmux 或任何 keepalive 占卡**，包括"只占一下"。
  entrypoint 必须是有界前台 driver（`bash -lc 'set -euo pipefail; ...; exec ...'`），工作结束即释放配额。
- 探针也必须是有界真实工作并立即退出。
- 长任务用 tmux + 显式日志与 `.exit` 标记，报告产物路径与 ETA，确认在推进后**停止盯守**，不要轮询。

## Definition of done

全部满足才报告"完成"：

1. 用户要求的结果已实现；
2. 跑过**与改动规模相称**的检查——纯文档改动不跑训练测试；
3. 如实说明结果与失败，包括**没有验证的部分**；不要用"应该没问题"代替"未验证"；
4. 改了行为、契约或数据结构时，拥有该事实的文档已在同一次改动里更新；
5. 非平凡决策已按 `docs/decisions/README.md` 记录。

不要交完第一版实现就停下来等审阅——继续做到上面全部满足为止。
