# 固定回放预算下的语言模型领域适应与遗忘

本目录基于原始[开题报告](开题报告.pdf)开展24小时研究。当前工作进行中，未产生的实验结论不会作为结果报告。

- 研究时段：2026-11-06 03:26:32 至 2026-11-07 03:26:32 UTC。
- 主模型：Pythia-410M；目标域：FineMath finemath-4plus；替代通用回放：C4；通用评估：独立C4与WikiText-103。
- 实测硬件：4×H100 80GB。大型缓存/完整优化器checkpoint位于`/tmp/grad-research-20261106`。因共享盘曾写满，本目录当前链接到该临时盘工作树；已经核验的持久快照及主最终权重另存于同级`research_delivery/`，见下面交付记录。

## 导航

- [研究报告工作稿](研究报告.md)与[PDF预览](reports/研究报告_工作稿.pdf)
- [文献调研](literature/文献调研.md)
- [环境重建与移交](docs/环境重建与移交.md)
- [已校验的共享盘快照](docs/durable_snapshot_latest.json)
- [主24条件最终模型的共享盘无损权重包验收](docs/primary_weights_durable_publication.json)与[恢复说明](docs/无损权重交付格式.md)
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

随后用`reproduce --registry <新registry路径> --gpus 0 --deadline <新的UTC截止时间>`复核输入并打印启动命令；添加`--execute`后以detached scheduler训练新运行。先按冻结环境文件准备依赖，运行时版本和实际精度仍由每项summary记录。该helper及测试仅准备和dry-run验证，未执行重训。
