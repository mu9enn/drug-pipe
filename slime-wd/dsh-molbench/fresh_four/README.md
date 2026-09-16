# 四 setting 全新单次测评（2026-09-16）

目的：停止按历史科学得分保留/补跑的诊断实验，重新生成公平可追溯的单次轨迹。旧结果保留供审计，不进入新成绩。

任务固定：MS-1 50、MS-2 37、Mol-edit 39、Mol-opt 39，共每组 165、四组 660。完全不运行 MS-3。

| 队列 | 先执行 | 随后执行 |
|---|---|---|
| 1 | pretrained L1（orig-l1） | SFT L1（sft-l1） |
| 2 | pretrained hier（orig-hier） | SFT low-lr L1（lr2e6-l1） |

每路模型服务 2 张 GPU、TP=2、一次一题，合计最多两题并行。GPU 作业 priority=9，有限驱动，结束后释放 GPU。后继 setting 待本路完成后提交。

CPU 开发工作空间运行 DSH、runner、静默 MCP 适配器。MCP 路径为 CPU -> 实验室批准 HTTP 代理 -> SCP，完全不经 molclaw-relay。GPU 仅运行模型推理；CPU 通过既有平台 SSH 转发调用 GPU loopback 模型 API。

固定政策：

- 新结果目录，拒绝既有 run manifest，不读取历史答题结果、不做得分筛选。
- 每题 attempt=1，不启用 recover-infra、retry-infra-failed、continuation 或 resume。
- 保留工具内部带同一 jobId 的有限传输重试；不因工具报错重跑整个 agent。
- 每题预算 14,400 秒。失败、超时、缺失均保留在预先固定的分母中。
- 仅在尚无 run manifest 且没有任何题目 record 的启动阶段允许最多三次准备尝试；题目开始后框架中断则停止该队列，等待明确处理，不重跑已尝试题。
- 主评分为 fence_only，仅剥离唯一 JSON/无语言标签代码块，再按原科学规则计分。全部答案仅来自本 setting 新输出。
- 四组均附加报告剔除已知同源 QED 题后的 Mol-opt 38 题成绩，排除 ID 48b3a4aa-d573-44f9-8fda-dfc4757a3aaa。不会依据本次得分决定是否排除。

远端清单：
`/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug_wd/drug_pipe_regular_v1_20260908/experiments/fresh_four_0916d/manifest.json`

远端执行脚本：
`/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/reports/fresh_four_20260916/`

结果目录：
`/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/outputs/dsh_molbench_evals/aligned-v4-9b-<setting>-fresh-0916d/`

每组的主评分输出为 `scores_fence_only/`，附有逐题来源和 SHA256；辅助去重叠成绩为其下 `mo_opt_task_disjoint/`。

验证：四组真实 rjob dry-run；priority、配额、GPU/CPU/内存、镜像、挂载、CPU_FRAMEWORK、模型路径及关闭 EVAL_RECOVERY 校验通过；四组同一 165 个不重复任务 ID；源码哈希和 Python 语法检查通过；CPU runner 中无整题恢复/续跑选项。

旧的 av4-cpu-orig-hier-0916c-gpu2 已停止且 active=0；旧 SFT 作业此前已停止；旧 CPU 队列、runner 和框架已退出。没有停止无关训练作业。

## 提交确认（15:54 CST）

首轮框架准备遇到代理 CONNECT 502，在任何题目启动前失败；第二轮两路均已取得 adapter_ready。首批 GPU 作业已提交并由 rjob 确认 Inqueue：

- av4-fresh-orig-l1-0916d-gpu2
- av4-fresh-orig-hier-0916d-gpu2

排队事件为当前可匹配节点的 CPU/GPU/内存资源不足；尚未开始答题。SFT L1 与 low-lr L1 已进入各自后继队列，前一 setting 完成后自动提交，最多两题并行。启动准备次数不计为题目尝试；所有题目的首条轨迹仍从 attempt=1 开始。
