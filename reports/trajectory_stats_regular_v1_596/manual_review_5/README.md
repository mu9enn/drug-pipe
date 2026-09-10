# 五条不同长度的训练轨迹

选取 3、12、24、42、74 步，覆盖 PF、E2E、VS、AC、KG 五种任务。此为人工检查的跨度选样，不是随机样本或质量合格样本。

来源：`/mnt/shared-storage-user/sdpdev-fs/sunxiangyu/drug_wd/drug_pipe_regular_v1_20260908/training/qwen35_sft_train.jsonl`。SHA-256 已核对发布清单。

每个 JSON 都是训练文件中的完整原始记录，仅使用 indent=2、ensure_ascii=False 格式化；保留 messages、reasoning_content、tool_calls、工具返回、system、catalog 和 tools schema，不截断、不删改。导出后重新解析并与原记录做完整相等检查。

步数为 assistant 决策轮数，包含最终回答。JSON 字符串内的换行按 JSON 标准保留为 \n。

|文件|ID|任务|步数|工具调用数|完整 token 数|
|---|---|---|---:|---:|---:|
|[01_03steps_PF_react_pf_817cf3ecc8c8c7a9.json](01_03steps_PF_react_pf_817cf3ecc8c8c7a9.json)|react_pf_817cf3ecc8c8c7a9|PF|3|8|47,130|
|[02_12steps_E2E_react_e2e_58d688cf14a18f3e.json](02_12steps_E2E_react_e2e_58d688cf14a18f3e.json)|react_e2e_58d688cf14a18f3e|E2E|12|21|54,550|
|[03_24steps_VS_react_vs_f4c38a0c1c178b53.json](03_24steps_VS_react_vs_f4c38a0c1c178b53.json)|react_vs_f4c38a0c1c178b53|VS|24|77|92,648|
|[04_42steps_AC_react_ac_d11b14f514e93955.json](04_42steps_AC_react_ac_d11b14f514e93955.json)|react_ac_d11b14f514e93955|AC|42|54|74,059|
|[05_74steps_KG_react_kg_a844be168c9f1d8e.json](05_74steps_KG_react_kg_a844be168c9f1d8e.json)|react_kg_a844be168c9f1d8e|KG|74|73|125,155|
