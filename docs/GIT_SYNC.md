# Git管理与三处同步

GitHub `main` 是已发布源码的汇合点；H机和采集机可以各有尚未发布的修复，不能只用修改时间或“哪台通常更新”决定覆盖方向。

## 版本化边界

- 版本化：可复用源码、测试、接口契约、项目skill、可审阅的补丁和流程图。
- 留在机器本地：密钥、`.env`、MCP私有配置、provider数据库、运行日志、raw轨迹、数据release、checkpoint、模型、环境和依赖缓存。
- 旧批次runner与历史分析脚本先判断是否仍适用。未跟踪不等于该删除，也不等于应当一律`git add .`。
- `runtime/claude` 是现有机器配置的执行包装器，目前未在主仓版本化。它依赖的可复用 `pipeline.claude_agent.cleanup_invocation` 已纳入源码。Git同步不包含包装器、provider设置或运行中进程状态。

## 常规同步

1. 查看 `git status --short`、当前分支及每个dirty子模块。
2. `git fetch origin main` 后比较HEAD和**刚抓取的**origin/main；过时的tracking ref不能证明已最新。
3. 工作树干净且没有本地独有提交时，只做 `git merge --ff-only origin/main`。存在本地修复则先留存并逐项合并，不用`reset --hard`、强制checkout或`clean`清理运行环境。
4. 发布仅使用明确的 `git push origin main`。不要`push --all`或`--mirror`：本机可能保留含历史大文件的backup分支。
5. 两机核对 `HEAD^{tree}`、tracked工作树差异与子模块pin/补丁。源码一致不等于provider、数据或运行时全部一致。

建议仓库级设置：`pull.ff=only`、`push.default=simple`、`fetch.prune=true`，main跟踪origin/main。两机GitHub SSH地址使用 `git@github.com:mu9enn/drug-pipe.git`；连接权限仍以各机实际认证为准，不复制凭据。

## DeepSeek Harness

继续固定 `.gitmodules` 所列上游仓库及主仓gitlink版本。两机的readImage开关修改已完整收录在 `patches/deepseek-harness/optional-read-image.patch`，不需要一个尚未发布的私有子模块commit。

按该目录README判断补丁未应用或已应用；补丁造成的dirty状态是预期状态。只有超出受管补丁的差异才需要额外核对。不要在dirty子模块上直接强制更新或恢复上游最新master；上游更新不等于本项目已验证兼容。

## 本次同步包含与未覆盖的范围

本次将H机的测评生命周期/可选资源池修改与采集机的MCP初始化、答案恢复和DSML边界修复合并，并保留各自依赖的通用帮助文件。评测worker脚本继续保持可执行位。

运行目录中的一次性startup/续跑钩子不因此自动成为主仓通用启动流程。当前实际任务状态见 `STATUS.md`；流程审阅中应将“可从源码复现的逻辑”和“某次run内的临时部署”分开。
