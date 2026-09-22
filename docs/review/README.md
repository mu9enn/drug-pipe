# Drug-Pipe流程与意图审阅

先读 [简要审阅表](intent-review.zh.md)，再在Understand-Anything中选择“业务领域”视图。按五个领域、十条流程逐步展开；每一步都说明职责、已有保护和限制，并可定位源码。结构图提供八层和十步导览，需要时再下钻。

## 产物

- `.ua/domain-graph.json`：高层领域、流程与步骤。
- `.ua/knowledge-graph.json`：88个关键文件的结构、关键函数与关系。
- `.ua/analysis-scope.json`：实际覆盖范围、代码快照与校验说明。
- `.ua/fingerprints.json`、`meta.json`、`intermediate/scan-result.json`：后续更新基线。

使用 [Understand-Anything](https://github.com/Egonex-AI/Understand-Anything) 的 `understand`、`understand-domain` 和 `understand-dashboard` skills 完成。图为本次明确范围的源码快照，不等于全部vendor代码或所有历史数据已验收；图上的tested/tested_by只表示测试关联，不表示通过记录。

## 打开与更新

在已安装插件的Codex中对本仓库运行 `understand-dashboard`。页面需要启动器打印的完整token URL；默认高层领域视图适合本次审阅，完整结构可切回代码视图。

源码改变后运行 `understand` 更新，再运行 `understand-domain` 派生流程。`.ua/config.json`关闭了自动更新，没有安装commit hook。读取审阅表时，源码链接固定到当次集成commit；后续实现变化应更新对应结论，而不是仅更新图的时间戳。

本轮重点保留问题生成、真实采集、证据修订、SFT发布/训练、真实评测与收尾；通用Slime/DSH内部实现、机器私有运行脚本、raw、模型和checkpoint未全面展开。Git管理边界见 [GIT_SYNC.md](../GIT_SYNC.md)。
