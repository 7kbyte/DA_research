# C 实际生成回放库的输入诊断

这是已经冻结输入的事后描述性诊断，不据此更换库、方法或参数，也不追认其为预注册结果分析。本诊断只读取冻结 C registry/config 所指向的生成池、initial teacher statistics、实际 bank tensor 和小 manifest，不读取 C validation/test/behavior 结果，不调用教师或 GPU，不重新生成或选择样本。现有主 410M 的 12 行几何表已经足够覆盖 C410M 的同一批输入，因此复用其数值并交叉核对输入 SHA；新增计算只补齐 1B 与 2.8B 的实际规则。

完整表有 30 行：410M/1B 各 B4、D、M、C × 3 seed，2.8B 仅实际使用的 B4、M × 3 seed。2.8B 不添加未使用的 C 或 D 条件。表内每个 pool 为 1957 个 full-512 输入块、1,000,027 个预测 token；bank 为 196 块、100,156 个唯一预测 token。indices、doc_ids、token/mask、canonical chunk hashes 和容量逐一与实际冻结 tensor 核对；各库均 196 个唯一块。

| 模型 | 规则 | 库内 teacher NLL 均值 | NLL 经验 CDF 最大差 | 最近代表特征平方距离均值 |
|---|---|---:|---:|---:|
| pythia-410m | 随机 B4 | 3.2432 ± 0.0632 | 0.0635 ± 0.0277 | 0.2413 ± 0.0035 |
| pythia-410m | NLL 比例分层 M | 3.2585 ± 0.0086 | 0.0322 ± 0.0051 | 0.2387 ± 0.0067 |
| pythia-410m | 聚类代表 C | 3.0850 ± 0.0540 | 0.1068 ± 0.0218 | 0.2096 ± 0.0030 |
| pythia-1b | 随机 B4 | 2.9195 ± 0.0397 | 0.0675 ± 0.0070 | 0.3191 ± 0.0033 |
| pythia-1b | NLL 比例分层 M | 2.9254 ± 0.0215 | 0.0322 ± 0.0051 | 0.3220 ± 0.0038 |
| pythia-1b | 聚类代表 C | 2.7347 ± 0.0690 | 0.1186 ± 0.0244 | 0.2775 ± 0.0012 |
| pythia-2.8b | 随机 B4 | 2.6428 ± 0.0347 | 0.0558 ± 0.0190 | 0.3106 ± 0.0005 |
| pythia-2.8b | NLL 比例分层 M | 2.6946 ± 0.0110 | 0.0322 ± 0.0051 | 0.3116 ± 0.0075 |

以上 ± 是模型内三个 generation/selection seed 的描述性样本 SD（ddof=1），包含 pool 与 selection 的变化，不是补充 B 固定 pool/train seed 的十次选库 draw，也不是训练性能、bootstrap CI 或测试不确定性。CSV 另给每一库内部 196 块的 NLL 样本 SD；不要把两种 SD 混淆。410M 是主实验池/库的 byte-identical 复用，新增重复次数为零。

410M 与 1B 的三对 D/B4 均具有相同 selected indices、doc IDs、token/mask 及顺序，因此这些输入诊断相同。其 `.pt` 整文件 SHA 均不同，序列化文件身份与 tensor 内容身份不能混为一谈。D 行用于审计，并不增加独立样本数；这里不推导 D/B4 最终 checkpoint 是否相等，权重 alias 仍须另有逐对 final proof。

M 在每个模型内都更贴近原池 NLL 的经验 CDF；它没有同时保证更低的最近特征距离，1B 和 2.8B 的 M 均值略高于对应随机库。410M/1B 的聚类代表库降低了当前特征空间的平均最近代表距离，但 teacher NLL 分布偏向较低值、CDF 偏移更大。这些量描述规则改变了什么，不能推出保留能力或目标学习收益。

三个模型的 M 在同 seed 上选中的稳定 NLL 秩集合 SHA 逐一相同，因此 CDF 最大差也相同；这与共同 N/K、五层配额及同 seed 的秩采样机制一致，不代表 NLL 数值分布相同。各模型 teacher statistics/pool SHA、NLL 和特征维数均分别核验，特征维数为 1024/2048/2560。

最近距离沿用既有 analyze_bank_geometry.py：initial teacher 最后层有效预测上下文均值特征，先 L2 归一，再计算 pool 每点至最近 bank 向量的欧氏平方距离，最后汇总均值/p95。它不是语义覆盖率、梯度分层、局部 Fisher 方向覆盖或 heldout 分布覆盖率。特征空间随模型变化，距离只作模型内比较；绝不以跨模型绝对距离推断尺度因果规律。CDF 和 Wasserstein-1 均为描述量，不作独立样本检验或性能保证。

五个 NLL 层大小均为 [392,392,391,391,391]，整数配额为 [40,39,39,39,39]。固定 pool 条件下，对层内均匀无放回抽样取标签均值的解析期望，层权重总变差及期望 CDF 最大偏差均为 0.003775040931。最易的第一层因最大余数及稳定 tie-breaking 多分一个名额，九个 M pool 的期望 NLL 均值偏差介于 -0.005568 与 -0.005257 nats；这是固定 pool 的整数配额混合偏差，不是实际抽取误差、梯度偏差或 retention 效应。解析 CDF 期望与某次随机库的 CDF 距离也不能互换。

本次完整输入分析实测 wall 3.885005s、process CPU 5.177573s，torch/OMP/MKL/OpenBLAS 均限制 2 线程，未使用 GPU。计时范围为输入 hash/loading/核验与几何/解析计算，不含进程启动 import、CSV/文档写入；CPU 时间不是墙钟等待。此前三次开发执行的已测分析区域时长单列 timing_history.json，不能把本次区域计时宣称为全部工程时间。

[逐模型/seed 完整 CSV](../results/extended_bank_diagnostics/bank_geometry.csv)、[三 seed 描述性汇总 CSV](../results/extended_bank_diagnostics/bank_geometry_summary.csv)、[全部 selected indices、输入关系和 152 项源 SHA 的 JSON](../results/extended_bank_diagnostics/bank_geometry.json)、[CPU 小 wrapper](../scripts/analyze_extended_bank_geometry.py)。沿用[既有主410M诊断](回放库分布诊断.md)，没有再造同一批 410M 几何产物。
