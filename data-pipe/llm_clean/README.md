# LLM Clean

这个目录保存 LLM clean 流程所需的 prompt 和辅助脚本。

## 文件说明

- `run_llm_clean.py`：真正的批处理编排逻辑，包含逐样本执行、清理校验、`tqdm` 进度条和最终收集调用
- `prompt.md`：给 Claude Code 的逐文件清理提示词模板
- `collect_llm_cleaned_json.py`：把 `cc-workdir/*/*-cleaned.json` 收集到统一输出目录
- `scripts/run_llm_clean.sh`：完整第二阶段入口，执行 LLM semantic repair、final hard-clean 和最终 validator

## 入口脚本

用户入口在 `scripts/run_llm_clean.sh`：

```bash
bash scripts/run_llm_clean.sh /path/to/json_trajectory_dir
```

它会：

- 为每个源 JSON 创建独立的 `cc-workdir/<source_stem>/`
- 在本地 workdir 内调用 Claude Code
- 让 Claude 直接写出 `<source_stem>-cleaned.json`
- 再把所有 cleaned JSON 收集到 `<input_dir>/cleaned/`

## Prompt 变量

`prompt.md` 会在运行时注入以下占位符：

- `{{SOURCE_FILENAME}}`
- `{{SOURCE_STEM}}`
- `{{CLEANED_FILENAME}}`

prompt 会明确要求 Claude：

- 先复制源文件到 cleaned 文件
- 只编辑 cleaned 文件
- 不使用 MCP
- 不依赖 `.claude`
- 不输出 wrapper JSON

## 输出约定

cleaned 文件本身必须是完整的训练样本 JSON，顶层结构保持为：

```json
{
  "schema_version": "...",
  "id": "...",
  "messages": [...]
}
```

收集后的 LLM 原始输出目录是 `<input_dir>/cleaned/`。入口随后生成：

- `<input_dir>/cleaned_final/`：通过脚本2和最终 gate 的训练候选
- `<input_dir>/cleaned_final_reports/quarantine/`：仍有语义冲突的 LLM 输出
- `<input_dir>/cleaned_final_reports/quarantine_validator/`：validator 发现的其他 P0 invalid 输出
- `<input_dir>/cleaned_final_validation.{json,md}`：post-LLM 最终校验报告

`cleaned/` 不会被脚本2覆盖。
