# C完整结果解释框架（解封前，仅骨架）

仅据冻结合同、active registry、配置及分析schema整理；未读取C测试结果、partial C验证分数或行为结果。所有结果/排名栏保持“待完整门禁及ROOT授权后填”，不新增选择或统计规则。此处“补充C”指模型/数据/曝光扩展；方法名C仍指cluster选库。

## 固定范围与参照

| 模型 | 实际方法（每法seed11/22/33） | 训练run | final目标/回放预测tokens¹ | 回放λ；共享LR | NLL / 行为逻辑端点² |
|---|---|---:|---:|---|---:|
| 410M | B0/B1/B2/B3/B4/D/M/C | 24 | 100M / 5M | .05；3e−5 | 48 / 24 |
| 1B | B0/B1/B2/B3/B4/D/M/C | 24 | 100M / 5M | .05；3e−5 | 48 / 24 |
| 2.8B | B0/B1/B2/B4/M | 15 | 25M / 1.25M | .05；1e−5 | 30 / 15 |

¹ B0为target-only：b=λ=0、回放0；其余均b=.05。B1/B3为C4/generated CE，B2/B4为相应KL；D去重random、M分位NLL、C聚类，三者均generated KL。各模型LR/共享λ由此前完整seed101、2M **validation-only**校准冻结，不沿用A看过测试后的选择。2.8B全15run的25M为首个formal前纯算时fallback，原50M未作为该结果范围。有效batch均64 chunks（64×1/32×2/8×8）。

² NLL全final＋按target-validation选best，按模型**整个阶段共同门禁**，分析仍分两端点；行为仅final。逻辑数量不含各模型initial。D/B4仅凭双方实际checkpoint SHA绑定的完整model-state等价证明，才能省对应forward；不删除D逻辑端点或物理训练费，不假定best也等于final。

三模型共同使用`data_family=extended_v1`及`artifacts/data_extended`：target训练容量39,140个exact-distinct 512-token chunks，**20,000,540个prediction-token位置**（raw 20,039,680），不是20M不同词汇。NLL为matched extended初始模型参照：同model/revision＋heldout SHA＋数值/执行policy；FineMath/C4 test各979chunks/500,269预测tokens，WikiText529/270,319。410M也不能拿主`main_v3` NLL initial代用。NLL eval batch依次16/16/8，context512；FP32权重、BF16 autocast、strict及四SDPA/TF32等保持合同。

行为是另一范围：固定PiQA/HellaSwag、acc_norm主/acc辅、CC/CW/WC/WW分别报告；context2048、batch16/16/8。相同模型/题目及完整inference policy时可复用已有行为initial，不能把NLL baseline或不同模型baseline混用。

## 有效的预定配对

同model/data/b/λ/LR、同seed、同端点配对；final与best绝不混配。

| 范围 | NLL冻结六对中的有效项 | 行为冻结配对中的有效项 |
|---|---|---|
| 410M、1B | M−B4；C−B4；B2−B1；B4−B3；B3−B1；B4−B2 | M−B4；C−B4；B2−B1；B4−B3；D−B4 |
| 2.8B | M−B4；B2−B1；B4−B2 | M−B4；B2−B1 |

2.8B **没有B3、D、方法C**：对应对照记“不存在”，不是0或不显著；不能补全source×objective的2×2。D/B4的NLL等价/null证据单列，不擅自加进冻结六个统计配对。B0是target-only参照，本骨架不新增B0成对检验。差值方向为A−B：G更大有利A，F/KL更小有利A。

## 完整授权后逐项填充

1. **来源/覆盖**：填真实READY、plan、ROOT authority、first-read及COMPLETE SHA，完整logical/physical/alias/baseline数量；仅整阶段通过再填，不用partial表推出完整结论。记录每个best真实step、目标/回放曝光及与final是否相同，不能预填为100M/25M。
2. **每模型×端点方法表**：填G_new、F_C4、F_Wiki三个token指标及对应equal-document指标的三个seed个体、mean±sample SD（ddof=1）；附三域原始NLL/ppl、exact teacher||student KL及doc加权口径。G=initial−adapted目标NLL；F=adapted−initial旧域NLL。只填数值，不用测试重选λ、方法或最佳端点。
3. **配对表**：由同seed差值先求三seed mean/SD，保留run IDs、方向数及两侧实际曝光；best即使同名端点也可能是不同训练预算，须保留`equal_checkpoint_exposure`。对应`endpoints`、`endpoint_statistics`、`paired_effects`、`paired_statistics`，不将独立均值差冒充配对SD。
4. **权重与条件区间**：对照token加权与等文档结论是否一致；若不同，解释长文档/长度权重而非挑一种。`endpoint_document_bootstrap`/`document_bootstrap`只条件于固定checkpoints及可交换文档假设，不能替代三seed SD或称训练随机性CI。行为题目区间另表，同源活动/上下文依赖也限制IID解释。
5. **费用与结论句**：填unique actual_run_id全训练parent项、真实inference ID/alias及CPU分析实测；best不按曝光比例缩减已经发生的完整训练费，teacher子项不再加父项，共享准备不重复加。预留句：“在〔模型/端点〕，〔配对〕的三个seed差值为〔待填〕；token/doc〔一致或分歧待填〕，只支持该固定配置下的〔有限结论待填〕。”

**解释上限：** C是相关的稳健性扩展，不是独立总体重复。扩展target包含主训练前缀，部分heldout内容共享；跨模型还改变生成teacher/库内容、校准LR、microbatch，2.8B又是25M而非100M。与主410M/10M还同时改变训练容量、曝光与heldout family，不能当纯模型scale或纯时长的因果效应。best是validation选择政策，不是同预算final；三pipeline seeds不是三个预训练模型，也不能把两端点/别名/共享题目当额外重复。所有负结果、方法缺项及token/doc分歧均保留。

精确来源及配置逐项核对见[小收据](../results/C_interpretation_preparation_v1/receipt.json)；核心依据为[aux v2合同](supplemental_aux_analysis_protocol_v2.json)、三份active formal registry与对应formal protocol、[NLL范围](../configs/nll_scope_execution_freeze_v1.json)、[行为范围](../configs/behavioral_scope_execution_freeze_v2.json)及现有collector schema。本文件不授予解封或GPU权限。
