# Status

**唯一描述"当前状态"的文件。** 状态变化时只改这一份；其他文档提到状态只能写"见 `docs/STATUS.md`"。
字段为 `TODO` 表示**未知**——问用户，不要从其他文档推断（`MAINLINE.md` 与 `EXPERIMENT_ALIGNMENT.md`
就曾互相矛盾，静默择一是错的）。标 `(推断)` 的值来自可查证证据但未经所有者确认；与用户说法冲突时以用户为准。

Last updated: 2026-09-21

## 当前状态

| 字段 | 值 | 依据 |
|---|---|---|
| rjob / GPU 任务是否被授权 | 是，未暂停 `(推断)` | manifest `authorization` 字段；09-19/20 有 `Succeeded` rjob |
| 当前发布版本 | `drug_pipe_regular_v1_20260908`：596 行训练 / 87 题评测 / 112 题保留 / 88 tools `(推断)` | `experiments/experiment_manifest_0909a.json` |
| 发布根路径 | `/mnt/shared-storage-user/sdpdev-fs/sunxiangyu/drug_wd/` | 同上 |
| 激活的模型规模 | 仅 Qwen3.5-9B；更大模型矩阵暂停 `(推断)` | manifest `authorization`；job 名与评测分数键均为 9B |
| skill 环境 | 并行双臂 `l1-flat`(52) 与 `legacy-hierarchy`(68) | manifest `skills` 字段 |
| 当前活跃 run | `v9pre` 9B 系列（SFT + MS-1/2/3 评测） | `reports/`、rjob 列表 |

## 待处理的文档不一致

- `docs/MAINLINE.md` 首行"训练和 rjob 继续暂停"是 2026-09-08 的旧注记，与 9/9 授权及后续实际 rjob 活动不符。
- `docs/EXPERIMENT_ALIGNMENT.md` 引用的 `drug_pipe_regular_v1_20260908/...` 是**无根的相对路径**，仓库内解析不到。
- `docs/MAINLINE.md` 写"512 条完整训练轨迹"，当前 manifest 为 **596** 条。
