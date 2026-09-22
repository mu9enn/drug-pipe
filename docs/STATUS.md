# Status

**唯一描述"当前状态"的文件。** 状态变化时只改这一份；其他文档提到状态只能写"见 `docs/STATUS.md`"。
字段为 `TODO` 表示**未知**——问用户，不要从其他文档推断（`MAINLINE.md` 与 `EXPERIMENT_ALIGNMENT.md`
就曾互相矛盾，静默择一是错的）。标 `(推断)` 的值来自可查证证据但未经所有者确认；与用户说法冲突时以用户为准。

Last updated: 2026-09-22

## 当前状态

| 字段 | 值 | 依据 |
|---|---|---|
| rjob / GPU 任务是否被授权 | 是；本次 2 卡在线测评已明确授权 | 2026-09-22 用户指令 |
| 当前发布版本 | `v9-release-merged`：561 条训练，最终 561 步；本次评测 175 题 | 用户粘贴训练记录及 eval_five_0922a/manifest.json |
| 发布根路径 | `/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug_wd/v9-release-merged/` | 本次实物检查 |
| 激活的模型规模 | 仅 Qwen3.5-9B；更大模型矩阵暂停 `(推断)` | manifest `authorization`；job 名与评测分数键均为 9B |
| skill 环境 | 本次 v9 在线测评使用 `l1-flat`（52） | eval_five_0922a/manifest.json |
| 当前活跃 run | 本次 v9 测评已结束；175 条全部有终态（168 completed，7 failed），基础设施失败 0；原 RJob `av4-v9merged-l1-0922v9-gpu3` Succeeded，资源释放；两道续跑，原 173 条记录保持不变 | eval_five_0922a/runtime_repair_0922/final_verification.json |

## 待处理的文档不一致

- `docs/MAINLINE.md` 首行"训练和 rjob 继续暂停"是 2026-09-08 的旧注记，与 9/9 授权及后续实际 rjob 活动不符。
- `docs/EXPERIMENT_ALIGNMENT.md` 引用的 `drug_pipe_regular_v1_20260908/...` 是**无根的相对路径**，仓库内解析不到。
- `docs/MAINLINE.md` 写"512 条完整训练轨迹"，当前 manifest 为 **596** 条。
