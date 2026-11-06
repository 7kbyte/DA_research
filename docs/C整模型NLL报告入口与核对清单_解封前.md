# C整模型NLL报告入口与核对清单（解封前）

只审阅源码/身份元数据并运行合成测试，未打开任何C test/behavior结果或C训练summary。现有`analyze_supplemental_phase_v2.py analyze`已能产出完整C统计表，不另写collector或重复bootstrap。B独立review脚本硬绑定60个selection draws，**不能改标签后用于C**。本清单不签authority、不执行真实check/analyze；未来ROOT另行授权后按真实plan/SHA运行。

| 模型 | contract_id | 必须共同完成的逻辑端点 | 每端点方法×seed | 有效预定NLL配对数 |
|---|---|---:|---:|---:|
| 410M | `extended_pythia_410m_100m_nll` | 24 final＋24 best | 8×3 | 6 |
| 1B | `extended_pythia_1b_100m_nll` | 24 final＋24 best | 8×3 | 6 |
| 2.8B | `extended_pythia_2p8b_25m_nll` | 15 final＋15 best | 5×3 | 3 |

实际方法、b/λ/LR、曝光及缺项沿用[冻结解释框架](C完整结果解释框架_解封前.md)。每模型独立`role=extended/data_family=extended_v1`和phase_name；2.8B没有B3/D/方法C，不补0、不推断不存在的2×2。行为范围仅final，另行授权，本入口不读取。

## 运行与报告门禁

1. **原流程先完整解封。** 复用[C410原运行清单](C410_NLL解封前运行清单_20261106.md)和[aux-v2模板](../configs/supplemental_aux_analysis_commands_v2.json)。真实READY→带显式immutable cache policy的prepare→whole-phase check→精确ROOT TRUE authority→原analyze。缺一个best、baseline、alias或prior闭包都不能先读final。不要直接用`analyze_nll_v2.py --read-heldout-scores`绕过整模型/首次读取门禁，不用`--allow-partial`。
2. **报告读取再等该模型COMPLETE。** 仅在ROOT明确给出该模型`COMPLETE.json`路径/SHA后核验`complete=true`，其plan/authority/readiness/first-read SHA和身份一致，readiness全闭包ready；`endpoints`恰有final、best且`logical_conditions`分别24/24或15/15。随后才读该分析目录的CSV/JSON。两份`analysis_manifest.json`必须均complete、expected/completed数量正确、training_seed为重复单位、模型/phase/extended_v1/endpoint一致。缺COMPLETE或任一端点不完整即停止，不从partial文件拼表。
3. **参照必须实际matched initial。** 从同一冻结manifest的唯一baseline group确认model/revision、extended heldout SHA、数值policy、batch/context、scorer、initial来源/attestation；不能只凭`pythia-410m`名字复用主main_v3 initial。报告保留initial身份和来源SHA，不从另一个模型/数据family找更合适的参照。

原CPU分析命令仍为下列模板（本次不填真实authority、不执行；首次读取只能运行一次，不能换目录重置journal）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONPATH=src:scripts python scripts/analyze_supplemental_phase_v2.py analyze \
  --plan '<该模型真实PLAN>' --plan-sha256 '<PLAN_SHA>' \
  --authority '<ROOT明确TRUE_AUTHORITY>' --authority-sha256 '<AUTHORITY_SHA>' \
  --state '<含完整闭包的实际STATE>' --output '<新的该模型分析目录>'
```

外层完整wall/user/sys计时采用原运行清单的Bash `time`封装，包含内部SHA/preflight/统计/写表；guard子时长不再相加。独立后续报告审阅/作图另计父级时间，不用first-read到generated_at的calendar间隔冒充完整process耗时。

## 完整后直接使用的表与核数

final、best分别读取，不重跑统计：

- `endpoints.csv`：24或15行，全部run、三seed、六指标、实际target/replay与full-training曝光、baseline和alias身份。`endpoint_statistics.csv`中六指标各方法n=3、expected_n=3/complete，mean及`std_ddof1`；保留其他原始NLL/KL列，不用PPL或新综合分数选方法。
- 六项指标固定为`g_new/f_old_c4/f_old_wikitext`及对应`_equal_document`。每端点六指标统计为48或30行；完整collector还保留其他原始NLL/KL统计，不能误把全文件只应有48行。G越大越好，F越小越好，单位nats。
- `paired_effects.csv`：410M/1B每端点18行，2.8B 9行；按同seed先A−B，再由`paired_statistics.csv`取三差值mean±sampleSD，六指标对应36或18条统计。410M/1B为M−B4、C−B4、B2−B1、B4−B3、B3−B1、B4−B2；2.8B仅M−B4、B2−B1、B4−B2。D/B4 null另报完整state证据，不擅自加入第七统计配对。
- `endpoint_document_bootstrap.json`和`document_bootstrap.json`：保留每seed的token/doc条件区间，后者raw NLL A−B的目标域方向需转成G差值时取负并交换区间端点；不将三seed均值SD当区间，也不把文档bootstrap当训练随机性CI。token/doc若结论分歧，两者均报告并解释长度权重，不能挑一种。
- `actual_training_costs.csv`按`actual_run_id`跨final/best全局去重，只计一次完整训练；best实际曝光小，不按比例折扣已发生费用。`physical_inference_costs.csv`按canonical inference ID跨端点/复用去重，alias不加forward；完整scheduler占槽wall、worker内scorer/SHA及teacher子项不能双加。未测后处理/等待不写0。

现有`nll_v2_analysis.plot_frontier(bundle, out, figure_width=7)`可直接复用已核CSV构造的bundle，不必重新collect/bootstrap；final/best分图，全部seed点＋mean±sampleSD，PNG/PDF/SVG配CSV。通用CLI默认10in，不直接用该默认图作正文。报告还需逐seed配对/实际best曝光表；图不能代替这些表。无需为解封前占位图创建新pipeline。

解释只支持该模型固定配置下的描述。best是target-validation选点政策，配对两侧也可能不同预算；不能与同预算final混称。三seed是pipeline随机来源，不是三份预训练模型，final/best或D别名不增加重复数。跨模型、主10M与C100M/25M还同时改变数据容量、生成库、校准LR、microbatch及heldout family，不能写纯scale/纯时长因果结论。

源码SHA、55项合成测试和手工六指标mean/SD/配对曝光算例见[准备收据目录](../results/C_nll_report_preparation_v1/)。未产生真实C报告数字、排名或新选择。
