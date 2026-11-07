# 固定回放预算下的语言模型领域适应与遗忘

本目录基于原始[开题报告](开题报告.pdf)开展24小时研究的交付工作树。已完成的实验结论、尚未验收的产物与未执行的可选项分别标注，不以配置存在代替实证完成。

- 研究时段：2026-11-06 03:26:32 至 2026-11-07 03:26:32 UTC。
- 主模型：Pythia-410M；目标域：FineMath finemath-4plus；替代通用回放：C4；通用评估：独立C4与WikiText-103。
- 实测硬件：4×H100 80GB。大型缓存/完整优化器checkpoint位于`/tmp/grad-research-20261106`。因共享盘曾写满，本目录当前链接到该临时盘工作树；已经核验的持久快照及主最终权重另存于同级`research_delivery/`，见下面交付记录。

## 导航

- [研究报告工作稿](研究报告.md)与[PDF预览](reports/研究报告_工作稿.pdf)
- [17:30答辩演示工作稿](presentations/答辩工作稿_20261106_1730/答辩演示工作稿.pdf)与[可编辑源及说明](presentations/答辩工作稿_20261106_1730/阅读与编辑说明.md)：该时点快照，1B／2.8B与部分行为评价当时尚未解封；最新完整结果以[研究报告](研究报告.md)为准。
- [论文写作与答辩提纲](docs/论文写作与答辩提纲.md)及[开题承诺与实际交付对照](docs/开题承诺与实际交付对照.md)
- [主实验独立NLL结果](docs/主NLL测试结果讨论.md)与[补充A完整权重敏感性结果](docs/补充A_NLL完整结果.md)
- [A梯度日志的事后描述性诊断](docs/A梯度裁剪_事后描述性诊断.md)：完整126次训练的裁剪阈值超出率与梯度范数，保留事后分析及因果解释边界。
- [补充B的完整独立测试结果](docs/补充B_完整60run测试结果解释.md)与[验证结果](docs/补充B_完整60run验证结果解释.md)：60条件均已完成，比较固定生成池和训练流下的选库波动。
- [410M／100M完整长程结果](docs/C410_100M_NLL完整结果与正文素材.md)：24项训练、48个final/best逻辑端点与完整配对；[final权重持久发布](docs/C410_final_weights_durable_publication.json)与[全部24项best权重持久发布](docs/C410_best_weights_durable_publication.json)（best包native基座自带；尚未另做单例durable恢复）。
- 1B与2.8B长程结果已写入报告第6.4/6.5节，独立复核见[C1B NLL](results/C1B_nll_independent_review_v1/independent_review_receipt.json)、[C1B行为](results/C1B_behavior_independent_review_v1/COMPLETE.json)、[C2.8B NLL](results/C2p8B_nll_independent_review_v1/independent_review_receipt.json)、[C2.8B行为](results/C2p8B_behavior_independent_review_v1/COMPLETE.json)；两模型的final权重已持久发布并完成1B单例CPU恢复；两模型的best包已在scratch完成定长编码与整包自检，但尚无对应的ROOT发布审阅适配器，因此未发布。
- [文献调研](literature/文献调研.md)与[三篇近邻工作的定位复核](docs/三篇近邻文献_实验定位复核备忘录_20261106.md)
- [环境重建与移交](docs/环境重建与移交.md)
- [交付文件与恢复容量指南](docs/交付文件与恢复容量指南.md)：已发布文件、基座依赖、按需恢复空间与操作顺序。
- [已校验的共享盘快照（2026-11-07 最终项目归档，3,093,828,104字节）](docs/durable_snapshot_latest.json)与[持久发布收据](docs/snapshot_final_durable_publication.json)；最终可编辑工作树已从该归档恢复并验收。
- 主36条件的final与best共72个逻辑端点均已持久保存：[best端点验收](docs/primary_best_weights_durable_publication.json)、[主比例24条件验收](docs/primary_weights_durable_publication.json)、[预算敏感性12条件验收](docs/budget_final_weights_durable_publication.json)及[恢复说明](docs/无损权重交付格式.md)；[按需恢复两个指定模型](docs/已交付权重恢复示例.md)
- 补充A全部126个λ网格final端点现已跨包持久覆盖：主包18项引用、[验证所选包15份新增权重](docs/A_selected_weights_durable_publication.json)，以及[其余93项native包发布验收](docs/A_remaining_final_weights_durable_publication.json)／[ROOT接受](artifacts/provenance/A_remaining_final_publication_actual_v1/ROOT_acceptance.json)；来源与容量见[交付指南](docs/交付文件与恢复容量指南.md)。
- 补充B全部60个final端点已持久保存（固定训练/pool seed11的选库变异）：[发布验收](docs/B_all_final_weights_durable_publication.json)、[包旁独立收据](/data/202610/7kb/research_delivery/B_all_final_weights_lossless_v1.tar.receipt.json)；native基座与恢复工具自带，已完成[固定draw1001/B4单例实际CPU恢复](docs/B固定首例native持久包实际恢复验收.md)；见[容量指南](docs/交付文件与恢复容量指南.md)。
- [研究路线与时间安排](docs/研究路线与时间安排.md)
- [工程接口与实验约定](docs/ENGINEERING_CONTRACT.md)
- `configs/`：冻结前pilot及正式实验配置。
- `artifacts/`：数据、回放库与来源manifest。
- `results/`：实验输出和统计图表。
- `logs/`：安装、下载、训练、调度与资源使用记录。
- `src/replay_study/`、`scripts/`、`tests/`：实现、命令行入口与正确性测试。

## 研究解释边界

本项目测量特定文本分布的下一词预测损失与输出漂移。数学文本损失下降不等于数学推理能力提高；C4也不是Pythia的原始预训练语料。固定样本库大小、累计回放token和实际计算成本分别计量。pilot、正式对照、补充探索将独立标记。

## 异地解包与路径重定位

将代码、数据和统计归档解包到新的workspace，并按`artifacts/assets/model_provenance.json`或`extended_model_provenance.json`中固定revision与SHA，在新scratch的`models/<模型名>-final/`准备基础模型。配置中的两个旧workspace前缀及scratch路径由独立helper重定位：

```bash
python scripts/reproduce_experiments.py relocate --registry configs/formal_deterministic_registry.json --input-manifest artifacts/provenance/formal_deterministic_input_snapshot.json --new-workspace /your/research --new-scratch /your/scratch --namespace replica01
```

`relocate`先核对原配置、冻结core、目标/回放/验证/测试数据及模型SHA，再生成新namespace的配置、registry、queue与`reproduction_of`来源记录；原结果保持原位，路径调整不记作新科学设置。扩展实验额外重复传入其matched extended protocol作为`--input-manifest`；可用`--run-id`选择单项。复制时清除旧日期deadline，可用`--deadline`指定新的操作截止时间。输入目录中的symlink需在交付机器可解析，归档应携带实际数据字节。

随后用`reproduce --registry <新registry路径> --gpus 0 --deadline <新的UTC截止时间>`复核输入并打印启动命令；添加`--execute`后以detached scheduler训练新运行。先按冻结环境文件准备依赖，运行时版本和实际精度仍由每项summary记录。

已从实际共享盘快照重新解包，逐文件核对3925个归档成员，并使用归档内原始helper完成B1/B2/B3/B4、seed11的路径重定位和启动命令dry-run；共34项输入（其中包括10份冻结科学源码及固定基座文件）均通过核验，见[归档复现QA](docs/归档复现QA演练.md)。这验证了归档内容与启动准备，尚未验证重新训练的权重一致性，也未完成全新容器的环境重建。
