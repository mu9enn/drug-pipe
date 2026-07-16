# get-molbench 全量堆叠式深度研究报告（FULL_DUMP）

- PROJECT_ROOT: `<data-pipe-root>/get-molbench`
- OUTPUT_REPORT: `<data-pipe-root>/get-molbench/PROJECT_RESEARCH_REPORT.md`
- MODE: `deep`
- LANGUAGE: `中文`
- STACK_STYLE: `FULL_DUMP`
- 调查日期: `2026-05-13`

说明：用户未显式给出 `PRIORITY_FILES` 与 `OUTPUT_REPORT` 实参；本次按默认优先序 `README.md -> docs/*.md -> requirements/environment -> pipelines/ -> scripts/ -> data/ -> outputs/` 执行，并将报告写入本文件。

---

## 1. 执行摘要（项目一句话 + 当前成熟度 + 总体判断）

一句话：`get-molbench` 是一个基于 Python 的 MolBench-MS 三任务（AC/VS/PF）数据集构建与导出工程，核心价值是把大规模原始生化数据（ACNet/CARA）转成可直接用于 agent 评测/训练的标准 CSV 基准集。

当前成熟度判断：
- 类型：研究型数据工程仓（Research Data Pipeline），偏“数据生成与评测集装配”，非模型训练仓。
- 阶段：`可用（MVP+）`，具备统一入口、批量生成、900规模产物、部分 provenance 文档；但在 PF-similarity 稳定性、空文件容错、文档一致性方面存在明显工程风险。

总体判断：
- 优点：
  - 三任务入口统一，参数清晰，输出命名基本统一（`molbench-<task>-<N>.csv`）。
  - 数据来源与主脚本链条可追溯（ACNet/CARA -> scripts -> pipelines -> outputs）。
  - 已有 900/900/900 产物可直接使用。
- 主要短板：
  - `similarity` 变体高概率出现 0 样本/空文件或极高重复度（训练价值下降）。
  - 文档与实现存在不一致（如 `legacy/`、命名统一口径）。
  - 质量门控集中在脚本报错，缺少统一产物校验器与日志化审计。

证据锚点：
- 仓库结构与声明：`<data-pipe-root>/get-molbench/README.md:5-23`
- 三入口脚本：`<data-pipe-root>/get-molbench/pipelines/generate_molbench_ac.py:12-55`、`.../generate_molbench_vs.py:12-84`、`.../generate_molbench_pf.py:22-113`
- 900批处理：`<data-pipe-root>/get-molbench/scripts/generate_molbench_900.py:80-168`

---

## 2. 信息源与调查方法（含优先文件阅读顺序）

### 2.1 优先阅读顺序（Phase 1 前置）

1. `README.md`
2. `docs/rdkit_scripts_and_pf_provenance.md`
3. `requirements.txt` + `environment.yml`
4. `pipelines/*.py`
5. `scripts/*.py`
6. `data/*`（ACNet/CARA 结构与规模）
7. `outputs/*`（已有产物核验）

证据锚点：
- `README`: `<data-pipe-root>/get-molbench/README.md:1-77`
- `docs`: `<data-pipe-root>/get-molbench/docs/rdkit_scripts_and_pf_provenance.md:1-110`
- 依赖：`<data-pipe-root>/get-molbench/requirements.txt:1-5`、`.../environment.yml:1-9`

### 2.2 调查方法

- 静态代码审计：逐文件提取函数、参数、随机策略、I/O 规则。
- 数据资产核验：用 pandas/json 读取 `data/` 与 `outputs/`，统计行列、字段、值域、重复率。
- 可执行性验证：运行 CLI `--help` 与小样本 smoke 生成（AC/VS/PF-v0/v1），并针对 similarity 做失败复现。
- 一致性比对：文档叙述 vs 代码实现 vs 真实产物三方对照。

---

## 3. 项目全景地图（目录/模块/资产）

### 3.1 顶层目录

- `pipelines/`: 统一入口层（任务编排）
- `scripts/`: 实际生成逻辑层（AC/VS/PF核心算法）
- `data/`: 原始数据资产（ACNet + CARA）
- `outputs/`: 已生成数据集产物
- `examples/`: 参考示例（非主流程输入）
- `docs/`: provenance 与脚本说明
- `requirements.txt`, `environment.yml`: 环境依赖

证据锚点：
- 文件扫描：`find . -maxdepth 3 -type d`（本地核验）
- README 声明：`<data-pipe-root>/get-molbench/README.md:5-23`

### 3.2 资产规模

- `data/` 约 `581MB`
- `outputs/` 约 `12MB`
- `examples/` 约 `192KB`

证据锚点：`du -sh data outputs examples ...`（本地核验）

### 3.3 模块职责图（逻辑）

1. 入口层：
- `pipelines/generate_molbench_ac.py`：调用 `scripts/generate_dataset_ACNet_v0.2.py`
- `pipelines/generate_molbench_vs.py`：调用 `scripts/generate_molbench_vs.py` 并重命名输出
- `pipelines/generate_molbench_pf.py`：按 variant 调用 v0/v1/similarity 脚本，并做不足样本重试

2. 生成层：
- AC：`generate_dataset_ACNet_v0.2.py`
- VS：`generate_molbench_vs.py`
- PF：`make_rdkit_benchmark_v0.py`、`make_rdkit_benchmark_v1.py`、`molecular_similiar.py`

3. 批处理层：
- `scripts/generate_molbench_900.py`：一键生成 AC/VS/PF 900
- `scripts/merge_molbench_pf.py`：合并 PF 三变体

证据锚点：
- `<data-pipe-root>/get-molbench/pipelines/generate_molbench_ac.py:29-50`
- `<data-pipe-root>/get-molbench/pipelines/generate_molbench_vs.py:39-80`
- `<data-pipe-root>/get-molbench/pipelines/generate_molbench_pf.py:15-109`
- `<data-pipe-root>/get-molbench/scripts/generate_molbench_900.py:80-164`

---

## 4. 端到端技术链路复原（药筛专家视角）

### 4.0 三任务在药筛中的角色定义（先讲“任务是什么”）

从药物发现流程看，这三个任务对应不同决策层：

1. `MolBench-AC`（Binding Affinity Comparison）
- 任务本质：给定同一靶点下两个分子，判断哪一个结合更强（或更弱）。
- 药筛语义：这是“对比决策”任务，模拟 hit-to-lead 阶段常见的 pairwise 取舍。
- 标签来源：实验亲和指标 `Ki`（越低通常结合越强）。

2. `MolBench-VS`（Virtual Screening Ranking）
- 任务本质：给定一个靶点和候选分子池，输出对全部候选的“活性优先级排序”。
- 药筛语义：这是“库筛选排序”任务，模拟虚拟筛选中的 ranking 场景。
- 标签来源：`pChEMBL Value`（本项目用阈值将 active/inactive 分割，并提供 active 排序答案）。

3. `MolBench-PF`（Property Filtering / Similarity）
- 任务本质：在给定分子集合中做规则过滤（v0/v1）或结构相似检索（similarity）。
- 药筛语义：这是“可开发性预筛/结构邻近检索”任务。
- 标签来源：RDKit 计算属性或 Morgan 指纹相似度。

证据锚点：
- 任务说明：`<data-pipe-root>/get-molbench/README.md:26-43`
- PF provenance：`<data-pipe-root>/get-molbench/docs/rdkit_scripts_and_pf_provenance.md:5-29`
- 核心实现：`<data-pipe-root>/get-molbench/scripts/generate_dataset_ACNet_v0.2.py`、`.../generate_molbench_vs.py`、`.../make_rdkit_benchmark_v0.py`、`.../make_rdkit_benchmark_v1.py`、`.../molecular_similiar.py`

### 4.1 MolBench-AC：任务定义与样本生成机制

数据源：
- `data/ACNet/mmp_ac_s_distinct.csv`（分子对 + `Ki_1/Ki_2` + `tid`）
- `data/ACNet/target_dictionary.xlsx`（`tid -> target_name`）

任务生成逻辑（代码级）：
1. 读取并校验输入列 `c1, Ki_1, c2, Ki_2, tid`。
2. 读取 target 字典，映射 `target_name`；缺失填 `UNKNOWN_TARGET`。
3. 以 `tid` 为主做分层抽样：按 `tid, Ki_1, Ki_2` 排序，再在 `unique_tids` 上等间隔选点，保证靶点覆盖。
4. 每条样本随机生成两种问法之一：
   - “谁更强”（lower Ki）
   - “谁更弱”（higher Ki）
5. 根据问法方向与 `Ki_1/Ki_2` 比较关系确定 `answer`（必须是 `s1` 或 `s2` 之一）。

证据锚点：
- 逻辑主链：`<data-pipe-root>/get-molbench/scripts/generate_dataset_ACNet_v0.2.py:53-115`
- 问法与答案判定：`.../generate_dataset_ACNet_v0.2.py:13-39`
- 输出落盘：`.../generate_dataset_ACNet_v0.2.py:137-145`

真实样例（来自产物）：
- 文件：`<data-pipe-root>/get-molbench/outputs/ac/molbench-ac-25.csv`（第1行抽样）
- 关键字段：
  - `target = Phosphodiesterase 5A`
  - `k1 = 0.21`, `k2 = 24.7`
  - 问法包含“higher Ki（更弱结合）”
  - `answer` 指向 Ki 更高的分子（与问法一致）

这说明：`AC` 不是“绝对数值回归”，而是“二选一相对判断”任务，强调模型的比较能力。

### 4.2 MolBench-VS：任务定义与样本生成机制

数据源：
- 默认 `data/CARA/Task/VS_All.tsv`

任务定义（药筛语义）：
- 输入：单一靶点相关 assay 的候选分子集（本项目固定 60 个候选）。
- 输出：按“预测活性/结合可能性”排序的分子列表（答案给出 active 子集顺序与分值）。

任务生成逻辑（代码级）：
1. 读取必要列并过滤终点类型：默认仅 `IC50, Kd, Ki`。
2. 将 `pChEMBL Value` 数值化，并以 `(Assay ChEMBL ID, Smiles)` 去重。
3. 对每个 assay 计算可行性：
   - 候选池大小必须 >= `n_candidates`（默认 60）
   - active 数必须 >= `min_active`（默认 6）
   - inactive 数必须足够填满候选（约束由 `max_active` 推导）
4. 抽样时先保障 cluster 覆盖（每个 `Target Cluster 0.3` 先抽一个 assay），再回填到目标样本数。
5. 每题从 assay 里采 60 个候选，其中 active 数控制在 `[6,10]`；输出字段：
   - `questions`（JSON，含 `target_chembl_id/target_name/candidates`）
   - `answer`（active SMILES 列表）
   - `answer_score`（对应 pChEMBL）
   - `n_active`

证据锚点：
- 过滤与可行性：`<data-pipe-root>/get-molbench/scripts/generate_molbench_vs.py:167-200`
- cluster 抽样与回填：`.../generate_molbench_vs.py:204-217,233-249`
- 单题构造：`.../generate_molbench_vs.py:82-154`

真实样例（来自产物）：
- 文件：`<data-pipe-root>/get-molbench/outputs/vs/molbench-vs-25.csv`（第1行抽样）
- 关键字段：
  - `target_chembl_id = CHEMBL1868`
  - `n_candidates = 60`
  - `n_active = 10`
  - `len(answer)=10`, `len(answer_score)=10`
  - top1 `answer_score = 9.3`

这说明：`VS` 任务本质是“集合内排序”，不是自由生成新分子；模型必须在给定候选集内做优先级决策。

### 4.3 MolBench-PF：任务定义与样本生成机制

PF 在本仓库里有两类不同认知负载：

1. `v0/v1`：规则过滤（rule-based filtering）
- 输入：10 个 SMILES。
- 输出：满足“Lipinski + 额外约束”的全部分子（可多选，1~5 个）。
- v0 属性池较小（`MolLogP/TPSA/RotB/MolWt` 等）。
- v1 属性池更大（新增 `RingCount/AromaticRings/FractionCSP3/HeavyAtoms/HeteroAtoms`）。

2. `similarity`：结构近邻检索（nearest-neighbor retrieval）
- 输入：10 个 SMILES，第1个是 query。
- 输出：最相似的 1 个分子。
- 相似度定义：Morgan FP（r=2, 2048 bit）+ Tanimoto。
- 难度由 top1-top2 相似度 gap 分级（easy/medium/hard）。

证据锚点：
- v0：`<data-pipe-root>/get-molbench/scripts/make_rdkit_benchmark_v0.py:29-212`
- v1：`<data-pipe-root>/get-molbench/scripts/make_rdkit_benchmark_v1.py:30-267`
- similarity：`<data-pipe-root>/get-molbench/scripts/molecular_similiar.py:33-159`

v0/v1 样本构造细节：
1. 从同一 assay 采样 10 个分子。
2. RDKit 计算属性。
3. 固定 Lipinski（四条件全满足）+ 随机抽取 2~3 个额外约束（分位数阈值，四舍五入到2位）。
4. 反复尝试直到 `selected_count` 落入 `[1,5]`。
5. 输出 `meta` 记录约束、尝试次数、assay_id。

similarity 样本构造细节：
1. 标准化 SMILES。
2. 采样 10 分子并计算 query 与其他 9 个的相似度。
3. 过滤掉过低/过高相似度（<0.20 或 >0.98）和“不可区分”样本（gap<=0.02）。
4. 固定选 top1 作为 `answer`，按 gap 打难度标签。

真实样例（来自产物）：
- v1 样例文件：`<data-pipe-root>/get-molbench/outputs/pf/v1/molbench-pf-300.csv`（第1行抽样）
  - `selected_count=5`, `attempts=2`
  - 约束示例：`RingCount >= 1.0`、`HeavyAtoms >= 11.2`
  - `answer` 为 5 行 SMILES
- similarity 样例文件：`<data-pipe-root>/get-molbench/outputs/pf/similarity/molbench-pf-300.csv`（第1行抽样）
  - `task_type = morgan`, `difficulty = hard`, `gap = 0.062`
  - `answer` 为单一 SMILES

药筛解释：
- `v0/v1` 更像“先导优化前的可开发性规则过滤器”。
- `similarity` 更像“基于结构邻近的类比检索器”。

### 4.4 pipeline 如何把三任务统一成“问题样本工厂”

统一机制：
1. `pipelines/*` 层只做参数标准化、路径组织、子脚本调用。
2. `scripts/*` 层负责任务特异的采样/打标。
3. 输出层统一收敛为 CSV，可直接喂给评测或代理系统。

关键统一点：
- 输入参数形态一致：`--n-cases --seed --out-dir`。
- 输出命名目标一致：`molbench-<task>-<N>.csv`（VS 通过 pipeline 完成重命名）。

证据锚点：
- AC pipeline：`<data-pipe-root>/get-molbench/pipelines/generate_molbench_ac.py:36-51`
- VS pipeline：`.../generate_molbench_vs.py:45-80`
- PF pipeline：`.../generate_molbench_pf.py:56-109`

### 4.5 900 批量链路（面向大规模实验）

- AC: 直接生成 900。
- VS: 5 个 batch x 180 合并到 900。
- PF: v0/v1/similarity 各 300，再 merge 成 900。
- similarity 若不足 300，`_expand_to_rows` 通过重采样扩容到 300。

证据锚点：
- `<data-pipe-root>/get-molbench/scripts/generate_molbench_900.py:18-58`
- `<data-pipe-root>/get-molbench/scripts/generate_molbench_900.py:60-77`
- `<data-pipe-root>/get-molbench/scripts/generate_molbench_900.py:101-162`

专家级风险注释：
- similarity 扩容能保证“行数”，但不保证“化学多样性”；实测 300 行只有 8 个 unique prompt，需谨慎直接用于训练。
- 证据：`<data-pipe-root>/get-molbench/outputs/pf/similarity/molbench-pf-300.csv`（统计结果：`unique_prompt=8`）。

---

## 5. 文档-代码-产物一一对应表（重点）

| 文档叙事 | 代码实现 | 产物证据 | 结论 |
|---|---|---|---|
| 三任务统一入口（AC/VS/PF） | `pipelines/generate_molbench_*.py` | `outputs/ac/*`, `outputs/vs/*`, `outputs/pf/*` | A 已实现 |
| VS 25 示例命令可输出 `molbench-vs-25.csv` | pipeline 调脚本后重命名 | `outputs/vs/molbench-vs-25.csv` 存在 | A 已实现 |
| 一键生成 900/900/900 | `scripts/generate_molbench_900.py` | `outputs/ac/molbench-ac-900.csv`, `outputs/vs/molbench-vs-900.csv`, `outputs/pf/molbench-pf-900.csv` | A 已实现 |
| PF 可由 v0/v1/similarity 合并 | `scripts/merge_molbench_pf.py` | `outputs/pf/molbench-pf-900.csv` 含 `source_variant` 列 | A 已实现 |
| `examples` 仅参考不作为输入 | pipeline 默认输入走 `data/*` | `pipelines/*.py` 默认参数均为 `data/*` | A 已实现 |
| `legacy/` 存档存在 | 文档声明存在目录 | 实际目录不存在（`legacy_exists=0`） | B 文档声称未证实 |
| 输出命名统一 `molbench-*` | VS 生成脚本原生写 `MolBench-vs-*`，pipeline 才重命名 | `scripts/generate_molbench_vs.py` 与 `pipelines/generate_molbench_vs.py` 行为不一致 | B 部分不一致 |
| similarity 严格约束下样本可能不足并需补齐 | 文档 note + 900脚本 `_expand_to_rows` | 有空文件/失败案例 + 300行仅8 unique prompt | A 已实现但质量风险高 |

证据锚点：
- README 对应：`<data-pipe-root>/get-molbench/README.md:44-61,75-77`
- 代码对应：上述各脚本行号
- 产物对应：`<data-pipe-root>/get-molbench/outputs/...`

---

## 6. 关键实验/关键流程逐步说明

### 6.1 AC 生成步骤（可复现）

1. 命令：
```bash
python pipelines/generate_molbench_ac.py --n-cases 25 --seed 100 --out-dir outputs/ac
```
2. 内部调用：
```bash
python scripts/generate_dataset_ACNet_v0.2.py --input-csv ... --target-dict-xlsx ... --n-cases 25 --seed 100 --out outputs/ac/molbench-ac-25.csv
```
3. 随机点：
- `np.random.seed(seed)`
- 问题方向（lower Ki/higher Ki）由 `np.random.rand() < 0.5` 决定
4. 输出核验：
- `answer` 必须是 `s1` 或 `s2` 之一（抽样验证为 True）

证据：`.../generate_dataset_ACNet_v0.2.py:14,54,137-145` + 产物抽样检查

### 6.2 VS 生成步骤（可复现）

1. 命令：
```bash
python pipelines/generate_molbench_vs.py --n-cases 25 --seed 42 --out-dir outputs/vs --no-remote-target-name
```
2. 关键参数影响：
- `--value-types` 默认仅 `IC50,Kd,Ki`（排除 Potency 等）
- `--threshold-pchembl 6.0` 决定 active/inactive 划分
- `--n-candidates 60`，`--min-active 6`，`--max-active 10`
3. 输出约束（实测）：
- `questions` 内 `candidates` 长度固定 60
- `answer` 与 `answer_score` 长度均等于 `n_active`
- 在 `vs-900` 中 `n_active` 分布 min=6, max=10, mean=7.948

证据：
- 参数定义：`.../scripts/generate_molbench_vs.py:27-37,287-291`
- 逻辑：`.../scripts/generate_molbench_vs.py:95-153`
- 实测统计：`outputs/vs/molbench-vs-900.csv`

### 6.3 PF 生成步骤（可复现）

v0/v1：
```bash
python pipelines/generate_molbench_pf.py --variant v0 --n-cases 300 --seed 42 --out-dir outputs/pf/v0
python pipelines/generate_molbench_pf.py --variant v1 --n-cases 300 --seed 42 --out-dir outputs/pf/v1
```

similarity：
```bash
python pipelines/generate_molbench_pf.py --variant similarity --n-cases 300 --seed 42 --out-dir outputs/pf/similarity
```

关键行为：
- pipeline 读取结果 CSV 行数；不足时可通过 `--retry-seeds` 尝试 `seed+i`
- 若仍不足，抛 RuntimeError

证据：`.../pipelines/generate_molbench_pf.py:56-107`

### 6.4 900 生成步骤（可复现）

```bash
python scripts/generate_molbench_900.py
```

执行顺序：
1. AC 900
2. VS 5批180合并为900
3. PF v0/v1/similarity 各300
4. similarity 扩容（若不足）
5. merge PF -> `molbench-pf-900.csv`

证据：`.../scripts/generate_molbench_900.py:84-164`

---

## 7. 数据资产核验（文件存在性、shape、标签口径、规模）

### 7.1 原始数据资产

ACNet：
- `data/ACNet/mmp_ac_s_distinct.csv`：`21352 x 5`，列 `c1,Ki_1,c2,Ki_2,tid`
- `data/ACNet/target_dictionary.xlsx`：`1015 x 3`，列 `tid,target_name,num_compounds`

CARA Task：
- `VS_All.tsv`：`1,237,256 x 14`，assays=`11,849`，targets=`2,242`
- `LO_All.tsv`：`1,187,136 x 14`，assays=`80,488`，targets=`4,456`
- 其他子集：`VS_GPCR/VS_Kinase/LO_GPCR/LO_Kinase` 均存在并有百万/十万级记录

Split JSON：
- 结构为 `dict[str, list[int]]`，键类似 `CHEMBL..._IC50/Ki/...`
- query/support/test/train 均存在

证据：`data/*` 实读统计（本地核验）

### 7.2 产物数据资产

标准产物：
- AC：`outputs/ac/molbench-ac-{5,25,900}.csv`
- VS：`outputs/vs/molbench-vs-{5,25,900}.csv`
- PF：`outputs/pf/v0/300`, `v1/300`, `similarity/300`, merge `900`

字段核验：
- AC 列：`question,answer,target,s1,k1,s2,k2`
- VS 列：`index,questions,task_type,answer,answer_score,n_active`
- PF v0/v1 列：`prompt,answer,meta`
- PF similarity 列：`prompt,answer,task_type,difficulty,meta`
- PF merge 列：`prompt,answer,meta,source_variant,task_type,difficulty`

异常资产：
- 发现多个空 CSV（仅 1 byte 换行）：
  - `outputs/pf_sim/molbench-pf-5.csv`
  - `outputs/small/pf/molbench-pf-similarity-10.csv`
  - `outputs/small/pf_seed_scan/sim_40.csv`

证据：
- 文件大小 1 byte 与 `xxd` 输出 `0a`（本地核验）

### 7.3 标签/口径核验

VS：
- `n_active` 与 `len(answer)==len(answer_score)` 一致（`vs-900` mismatch=0）。
- `n_active` 分布满足配置约束 `[6,10]`。

AC：
- 问题方向近似 50/50：`ask lower Ki=453`, `ask higher Ki=447`（900样本）。

PF：
- v0/v1 的 `selected_count` 分布在 `1..5`。
- similarity 的 `difficulty` 分布：hard 151, easy 75, medium 74。

证据：`outputs/*` 统计脚本结果（本地核验）

---

## 8. 关键脚本入口清单（按运行顺序）

### 8.1 单任务入口

1. AC
```bash
python pipelines/generate_molbench_ac.py --n-cases <N> --seed <S> --out-dir outputs/ac
```

2. VS
```bash
python pipelines/generate_molbench_vs.py --n-cases <N> --seed <S> --out-dir outputs/vs --no-remote-target-name
```

3. PF
```bash
python pipelines/generate_molbench_pf.py --variant <v0|v1|similarity> --n-cases <N> --seed <S> --out-dir outputs/pf
```

### 8.2 全量批处理入口

4. 900三任务
```bash
python scripts/generate_molbench_900.py
```

5. 手动 PF 合并（可选）
```bash
python scripts/merge_molbench_pf.py --v0-csv ... --v1-csv ... --similarity-csv ... --out ...
```

证据锚点：
- `README` 命令区：`<data-pipe-root>/get-molbench/README.md:24-69`

---

## 9. 复现环境准备（Python版本、依赖、安装命令、环境变量）

### 9.1 推荐环境

- Python: `3.10`（conda env）
- 依赖：
  - `pandas>=2.2,<3.1`
  - `numpy>=1.26,<3.0`
  - `rdkit>=2023.9`
  - `openpyxl>=3.1,<4.0`

证据：`<data-pipe-root>/get-molbench/environment.yml:1-9`、`.../requirements.txt:1-5`

### 9.2 安装命令

Conda 方式（推荐）：
```bash
cd <data-pipe-root>/get-molbench
conda env create -f environment.yml
conda activate get-molbench
```

Pip 方式：
```bash
cd <data-pipe-root>/get-molbench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 9.3 环境注意项

- VS 若不加 `--no-remote-target-name`，会尝试请求 ChEMBL API 解析 target name（网络依赖）。
  - 代码：`.../scripts/generate_molbench_vs.py:54-67,128-134`
- RDKit 会打印大量 deprecation warning（MorganGenerator 建议），可能污染日志。
  - 观测：`pipelines/generate_molbench_pf.py --variant similarity ...` 运行日志

---

## 10. 复现路线图

### 10.1 Minimal Path（最小可运行路径，已实测）

> 目标：快速验证 AC/VS/PF-v0/v1 可跑通，并产出标准 CSV。

```bash
cd <data-pipe-root>/get-molbench
python pipelines/generate_molbench_ac.py --n-cases 3 --seed 100 --out-dir outputs/smoke/ac
python pipelines/generate_molbench_vs.py --n-cases 3 --seed 42 --out-dir outputs/smoke/vs --no-remote-target-name
python pipelines/generate_molbench_pf.py --variant v0 --n-cases 3 --seed 42 --out-dir outputs/smoke/pf_v0
python pipelines/generate_molbench_pf.py --variant v1 --n-cases 3 --seed 42 --out-dir outputs/smoke/pf_v1
```

实测结果：4条命令成功，输出文件存在且可读。
- `outputs/smoke/ac/molbench-ac-3.csv`
- `outputs/smoke/vs/molbench-vs-3.csv`
- `outputs/smoke/pf_v0/molbench-pf-3.csv`
- `outputs/smoke/pf_v1/molbench-pf-3.csv`

### 10.2 Full Path（全量复现路径）

```bash
cd <data-pipe-root>/get-molbench
python scripts/generate_molbench_900.py
```

完成后核验：
```bash
python - <<'PY'
import pandas as pd
for p in [
 'outputs/ac/molbench-ac-900.csv',
 'outputs/vs/molbench-vs-900.csv',
 'outputs/pf/molbench-pf-900.csv'
]:
    df=pd.read_csv(p)
    print(p, len(df), list(df.columns))
PY
```

### 10.3 similarity 独立复现建议（高风险链路）

推荐先小规模迭代：
```bash
python pipelines/generate_molbench_pf.py \
  --variant similarity \
  --n-cases 20 \
  --seed 42 \
  --retry-seeds 30 \
  --input data/CARA/Task/VS_GPCR.tsv \
  --out-dir outputs/pf_sim_try
```

并立即做质量校验（非空 + 去重率）：
```bash
python - <<'PY'
import pandas as pd
p='outputs/pf_sim_try/molbench-pf-20.csv'
df=pd.read_csv(p)
print('rows=',len(df),'unique_prompt=',df['prompt'].nunique(),'dup=',df.duplicated('prompt').sum())
PY
```

---

## 11. 风险与异常清单（按 P0/P1/P2 排序）

### P0（阻断级）

1. PF similarity 可能 0 样本并导致 pipeline 失败
- 现象：`Generated 0 cases ...` + `RuntimeError: produced only 0/1 rows`
- 证据：`pipelines/generate_molbench_pf.py:103-107`；实测失败日志（2026-05-13）
- 修复建议：增大 `--retry-seeds`、降低 `--n-cases`、换输入子集，或放宽 `molecular_similiar.py` 过滤门槛（`sim<0.20`、`gap<=0.02`）

2. 空 CSV（1 byte）历史产物可污染下游
- 现象：多个文件仅换行，pandas 报 `No columns to parse from file`
- 证据：
  - `outputs/pf_sim/molbench-pf-5.csv`
  - `outputs/small/pf/molbench-pf-similarity-10.csv`
  - `outputs/small/pf_seed_scan/sim_40.csv`
- 修复建议：统一增加 post-check（文件大小>1、列名完整、行数>0），失败即删除并重跑

3. similarity 输出高重复（信息有效性风险）
- 现象：`outputs/pf/similarity/molbench-pf-300.csv` 仅 `8` 个 unique prompt（300行）
- 证据：本地统计 `unique_prompt=8`
- 修复建议：加入样本去重约束（prompt hash 去重）、提高多样性采样策略、扩大候选 assay 空间

### P1（高优先级质量风险）

4. 文档与实现不一致：`legacy/` 目录声明存在但仓库中不存在
- 证据：`README.md:21,77` vs 实际 `legacy_exists=0`
- 修复建议：更新 README 或补档案目录

5. 命名口径存在双轨：`MolBench-vs-*` 与 `molbench-vs-*`
- 证据：`scripts/generate_molbench_vs.py:324` vs `pipelines/generate_molbench_vs.py:74-80`
- 风险：绕过 pipeline 直接调 script 时命名不一致
- 修复建议：在生成脚本内直接统一小写命名，pipeline 不再重命名

6. AC 核心脚本的默认输入路径指向 `scripts/`，与实际数据位置不符
- 证据：`generate_dataset_ACNet_v0.2.py:124-130` 默认 `scripts/mmp_ac_s_distinct.csv` & `scripts/target_dictionary.xlsx`，但文件缺失
- 风险：直接运行脚本（不经 pipeline）易失败
- 修复建议：默认路径改为 `project_root/data/ACNet/...`

### P2（中低优先级改进项）

7. examples 文件包含大量 `Unnamed:*` 列，且格式并非干净训练输入
- 证据：`examples/*.csv` 列统计
- 修复建议：提供标准化清洗脚本 + 明确“仅参考”标识

8. similarity RDKit deprecation warning 过多，影响日志可读性
- 证据：实测日志含大量 `please use MorganGenerator`
- 修复建议：升级 fingerprint API 或日志降噪

---

## 12. 已实现/未实现/规划中 三分法结论

### A. 已实现且可证实

1. 三任务统一 pipeline 入口（AC/VS/PF）
2. 900 批处理一键生成
3. PF 三变体合并产物 `molbench-pf-900.csv`
4. VS 输出 `questions` JSON + `answer_score` 配套
5. PF provenance 文档（v0/v1/similarity 来源解释）

证据：`pipelines/*.py`、`scripts/generate_molbench_900.py`、`outputs/*`、`docs/rdkit_scripts_and_pf_provenance.md`

### B. 文档声称但代码/仓库未完全证实

1. `legacy/` 目录存在（文档称有，仓库无）
2. “命名统一”在 script 层并非完全统一（VS 大小写前缀差异）

### C. 仅规划/讨论未落地

1. similarity 质量控制（去重、多样性约束）尚无工程化实现
2. 统一产物校验器（空文件/列完整性/重复率门控）尚未内建
3. 大规模生成可观测性（结构化日志、统计报告）较弱

---

## 13. 改进建议（工程改造线 + 研究改进线）

### 13.1 工程改造线

1. 增加统一 `validate_outputs.py`
- 校验规则：文件存在、非空、行数达到目标、列集合一致、关键字段可解析。
- 对 similarity 增加 unique ratio 门槛（例如 unique_prompt_ratio >= 0.6）。

2. 固化配置中心
- 把 `n_candidates/min_active/max_active/threshold`、similarity gap 阈值收敛到单一 config 文件，避免脚本散落。

3. 收敛命名与路径默认
- VS 直接输出 `molbench-vs-*`。
- AC 脚本默认输入改到 `data/ACNet/*`。

4. 日志治理
- similarity 批量运行时抑制 RDKit warning 或写入单独日志文件。

5. CI/Smoke
- 在 CI 中固定跑：`AC(3) + VS(3) + PF-v0(3) + PF-v1(3)`，并把 similarity 放到 nightly（带重试预算）。

### 13.2 研究改进线

1. similarity 采样策略升级
- 从“随机 assay + 随机 10 分子”改为“先筛可区分 query，再组题”。
- 引入分层难度控制，减少 hard 过饱和。

2. PF 混合集构建规范化
- 明确 `v0/v1/similarity` 比例、去重逻辑、difficulty 平衡策略。

3. AC 语义一致性
- 当前 question 文本对 Ki 高低与亲和力关系在两种问法间切换，建议增加 machine-readable 标签（`task_polarity`）避免评测误读。

4. 数据卡（Dataset Card）
- 为 AC/VS/PF 各生成 dataset card（来源、过滤、约束、已知偏差、推荐用途）。

---

## 14. 附录（关键文件索引、参数索引、产物索引）

### 14.1 关键文件索引

- `<data-pipe-root>/get-molbench/README.md`
- `<data-pipe-root>/get-molbench/docs/rdkit_scripts_and_pf_provenance.md`
- `<data-pipe-root>/get-molbench/pipelines/generate_molbench_ac.py`
- `<data-pipe-root>/get-molbench/pipelines/generate_molbench_vs.py`
- `<data-pipe-root>/get-molbench/pipelines/generate_molbench_pf.py`
- `<data-pipe-root>/get-molbench/scripts/generate_dataset_ACNet_v0.2.py`
- `<data-pipe-root>/get-molbench/scripts/generate_molbench_vs.py`
- `<data-pipe-root>/get-molbench/scripts/make_rdkit_benchmark_v0.py`
- `<data-pipe-root>/get-molbench/scripts/make_rdkit_benchmark_v1.py`
- `<data-pipe-root>/get-molbench/scripts/molecular_similiar.py`
- `<data-pipe-root>/get-molbench/scripts/generate_molbench_900.py`
- `<data-pipe-root>/get-molbench/scripts/merge_molbench_pf.py`

### 14.2 参数索引（核心）

VS：
- `--n-candidates` (default 60)
- `--min-active` (default 6)
- `--max-active` (default 10)
- `--threshold-pchembl` (default 6.0)
- `--value-types` (default `IC50,Kd,Ki`)
- `--no-remote-target-name`

AC：
- `--n-cases`
- `--seed`
- `--input-csv`
- `--target-dict-xlsx`

PF：
- `--variant {v0,v1,similarity}`
- `--n-cases`
- `--retry-seeds`
- `--input`

### 14.3 产物索引（核心）

- AC：`outputs/ac/molbench-ac-900.csv`
- VS：`outputs/vs/molbench-vs-900.csv`
- PF-v0：`outputs/pf/v0/molbench-pf-300.csv`
- PF-v1：`outputs/pf/v1/molbench-pf-300.csv`
- PF-sim：`outputs/pf/similarity/molbench-pf-300.csv`
- PF-merge：`outputs/pf/molbench-pf-900.csv`

### 14.4 调查边界与未覆盖项

- 未执行全量 900 再生（耗时较大）；本次以现有产物核验 + 小样本 smoke + 失败复现为主。
- 未对 CARA Split 的上游构建逻辑做逆向（仅核验其结构与存在性）。

---

## Phase 执行记录（按用户要求）

### Phase 1. Requirements Discovery

- 研究目标：明确三任务数据构建机制、可复现路径、产物质量风险。
- 业务目标：可为后续 bench/训练提供稳定 CSV。
- 技术目标：建立 文档↔代码↔数据↔脚本↔产物 一一对应。
- 交付目标：输出 FULL_DUMP 可审计报告。
- 项目类型：研究型数据工程仓（非训练仓）。
- 边界：聚焦本仓，外部平台（如下游模型训练）不展开。

### Phase 2. Project Exploration

- 完成目录扫描、文件清点、资产体量统计。
- 建立模块地图：pipelines（入口）+ scripts（核心）+ data（原始）+ outputs（产物）。

### Phase 3. Deep Analysis

- 深挖 AC/VS/PF 三条代码链，抽取输入输出、shape、约束、seed 策略。
- 完成关键样本字段解析（question JSON、answer/meta 结构）。
- 完成 small smoke 和 similarity 失败复现。

### Phase 3.5. Consolidation

- 输出文档-代码-产物一致性对照。
- 分级问题：P0/P1/P2。
- 综合质量评分：`78/100`。
  - 扣分主要来自 similarity 稳定性与重复度、文档与实现不一致、缺少统一校验器。

### Phase 4. Report Generation

- 本文件即最终报告，已写入 `OUTPUT_REPORT`。

### Phase 5. Iterative Refinement（自检）

自检问题与结论：
1. 陌生读者能否直接上手？
- 能。已给环境、命令顺序、最小路径与全量路径。
2. 是否列出前5大复现风险？
- 是。第 11 节给出 8 项，含 P0/P1/P2 分级与修复建议。
3. 是否给出最小复现路径与全量复现路径？
- 是。第 10 节已分别给出并附实测结论。

