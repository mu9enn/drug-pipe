# ToolRL：近期历史筛选主线

本次仍使用 605 条轨迹展开的 12,634 条原始 decision。不要因为后续 SFT 发布了另一批
596 条样本，就自动替换本次母数据。实际来源以 preparation_manifest.json 的路径和哈希为准。

## 三步流程

1. **准备比较副本。** 保留任务、用户补充要求、developer 约束、当前完整答案及当前工具定义；历史助手消息
   按完整消息从后往前装入 32K 编码预算。所有 tool-result（包括 skill 返回）都不进入编码文本。
   公共系统消息和工具全集另存并逐条校验内容完全相同；不同版本必须分开运行，不按“基本相似”删除。
   装不下的固定内容或最近一条完整助手消息继续保护。旧编码缓存不可复用到改变后的文本。
2. **运行官方筛选。** Qwen3-Embedding-8B 编码，NeMo 官方 K-means、组内比较与保留顺序不变。
   正式执行只传一个 eps；`cosine_sim_score >= 1 - eps` 是官方移除条件，不是固定保留比例。
3. **恢复训练数据。** 根据 ID 取回原始完整行，顺序不变。训练长度来自独立的 Qwen3.5 审计，绝不使用窗口长度替代。

比较条件只在第一步生成：有序调用清单（skill 身份、明确执行模式附在对应位置），以及前一结果的明确成功／错误／未知。
工具步骤不按任务类别隔离；最终回答保留任务类别边界。没有旧短描述、旧分类字段、配额或生成式摘要依赖。
成功／错误／未知仍从原始返回提取为比较边界，不把返回正文或状态描述拼入编码文本。
去掉返回会失去具体证据、skill 正文和错误细节，这是本轮明确接受的筛选近似；训练数据仍完整保留这些内容。
1～2 条的小组直接保留。窗口有损，所以“未入选”不能解释为严格重复或质量差。

## 执行入口

在 `slime-wd/slime` 下，使用已安装依赖的环境；`RUN` 是新的输出目录，`SOURCE` 是全量 decision JSONL。
编码模型和固定参数集中在 `drug_agent/toolrl/nemo_config.py`，GPU worker 必须运行真实前台任务并自动退出。
`RUN/venv` 和 `RUN/bootstrap` 应指向已经验证的隔离环境；launcher 在申请 GPU 前检查它们存在。
为防止已观察到的后端无报错卡住，编码阶段最多 4 小时、去重阶段最多 12 小时；超时退出释放 worker，
已完成缓存保留。超时不是“判为重复”或筛选完成。
vLLM 0.14.1 的满长请求在拆分调度时可能卡住，因此含恰好 32,768 tokens 输入的少数编码批次
逐条调用同一个编码器。不改变文本、模型配置或 NeMo 算法，仍逐个核对实际 token ID；普通批次不变。

```bash
python -m drug_agent.scripts.prepare_nemo_toolrl --input "$SOURCE" --output-root "$RUN/prepared"
bash drug_agent/scripts/launch_nemo_worker.sh "$RUN" smoke "$EPS"
# 检查小批真实相似样本，尤其新增的跨任务配对后，再执行：
bash drug_agent/scripts/launch_nemo_worker.sh "$RUN" full "$EPS"
```

`EPS` 必须显式提供；没有宣称适用于所有数据的默认阈值。需要阈值诊断时，可独立重复执行
`run_nemo_toolrl dedup --run-root "$RUN" --eps ... --smoke`，复用已保存的官方相似度文件；正式流程不自动扫四档。
同一运行目录修改文本或模型时会拒绝旧缓存。仅修改分组或阈值不会让编码缓存失效；每个缓存条目仍核对 ID、文本哈希和向量。

### 2026-09-11：保留 20%～30% 的阈值实验

用户指定以原始全量 decision 数为分母，目标保留 20%～30%。小批检查后运行：

```bash
bash drug_agent/scripts/launch_nemo_worker.sh "$RUN" experiment 0.05
```

前台任务依次完成全量编码、官方组内相似度计算、阈值标定，结束自动释放 GPU。
`sweep_nemo_thresholds` 复用官方分数检查宽范围 eps 网格和所有实际分数边界，选择保留数最接近
25% 的可行值，再调用官方 `IdentifyDuplicatesStage` 验证移除 ID。保护记录、小组直接保留记录
都计入分母和保留数；不拆同分样本、不改变比较边界，不额外抽样凑数。不可达时报告限制并退出。
阈值识别是 CPU 操作，不为不同 eps 重复申请 GPU、重新编码或重新聚类。
结果位于 `threshold_experiment/threshold_report.json`；数量达到目标不等于质量检查通过，
正式采用前还需检查阈值附近的完整样本对和工具、skill、任务覆盖。

长度统计是独立训练衔接，不是筛选算法的一部分。已有审计只有在源文件、统计文件哈希及模板配置一致时才能复用。
不能直接把旧报告中的源哈希改成新哈希。

```bash
python -m drug_agent.scripts.audit_toolrl_v8_lengths --input "$SOURCE" --model "$QWEN_MODEL" \
  --output-root "$LENGTH_ROOT" --no-pretty
python -m drug_agent.scripts.materialize_nemo_toolrl --run-root "$RUN" --eps "$EPS" \
  --length-details "$LENGTH_ROOT/length_details.jsonl" --length-report "$LENGTH_ROOT/length_report.json"
```

可额外传 `--qwen-source` 做逐条母轨迹历史前缀校验。正式交付还需运行
`validate_nemo_toolrl_delivery` 检查实际读取、原生解析、4 候选分组和训练长度。
本轮不修改奖励、读取顺序、生成预算，也不启动 RL。完整教师回答仍参与现有 32K／64K 分档，超长记录单列而不删除。

新代码通过测试不代表全量任务完成；只有正式去重、导出和读取验证报告落盘后，才能报告最终可用数量。
