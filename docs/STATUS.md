# 研究运行状态

更新时间：2026-11-07T01:42:26.625001+00:00
截止时间：2026-11-07 03:26:32 UTC（北京时间11:26）
已用 22.27 小时；剩余 1.73 小时。

以下状态来自运行日志。配置完成不等于实验完成；正式结论以最终研究报告为准。

当前阶段：全部267项正式训练与514项冻结独立评价完成，写入32页研究报告；十份权重包与最终项目快照已发布，可编辑树已恢复验收；三轮归档重训QA在最终交付归档上覆盖全部八个主组，16/16端点完整模型状态逐位一致。

旧非确定性pilot/formal仅供诊断；正式主结果为主strict36。补充训练按冻结计划在00:00 UTC结束科学计算，全部评价由后继调度器完成。研究交付工作稿见研究报告.md（31页）。

| 调度阶段 | 完成 | 运行 | 待执行 | 失败/阻塞 |
|---|---:|---:|---:|---:|
| formal_deterministic_job_state | 36 | 0 | 0 | 0 |
| formal_job_state（已隔离） | 9 | 0 | 27 | 0 |
| lambda_deterministic_v3_job_state | 17 | 0 | 0 | 0 |
| lambda_job_state（已隔离） | 16 | 0 | 0 | 0 |
| lambda_v3_job_state（已隔离） | 16 | 0 | 0 | 0 |
| lr_deterministic_v3_job_state | 4 | 0 | 0 | 0 |
| supplemental_job_state | 607 | 0 | 0 | 0 |
| v3_lr_eval_job_state（已隔离） | 4 | 0 | 0 | 0 |
| final_evaluation_successor（含299项继承） | 514 | 0 | 0 | 0 |

## 当前单一调度器内的阶段

以下为上述 supplemental 总行的分解，不另加为新的任务；按冻结 job ID 统计。

| 阶段 | 完成 | 运行 | 待执行/尚未登记状态 | 其他状态 |
|---|---:|---:|---:|---:|
| batch_benchmark | 2 | 0 | 0 | 0 |
| A_lambda | 108 | 0 | 0 | 0 |
| extended_v2_pythia_2p8b_m8_lr | 4 | 0 | 0 | 0 |
| B_conditional_bank_variance | 60 | 0 | 0 | 0 |
| extended_v2_pythia_410m_m64_lr | 4 | 0 | 0 | 0 |
| extended_v2_pythia_1b_m32_lr | 4 | 0 | 0 | 0 |
| extended_v2_pythia_410m_m64_lambda | 16 | 0 | 0 | 0 |
| extended_v2_pythia_1b_m32_lambda | 16 | 0 | 0 | 0 |
| extended_v2_pythia_2p8b_m8_lambda | 16 | 0 | 0 | 0 |
| primary_behavior_initial_timing | 1 | 0 | 0 | 0 |
| extended_v2_pythia_410m_m64_formal | 24 | 0 | 0 | 0 |
| extended_v2_pythia_1b_m32_formal | 24 | 0 | 0 | 0 |
| extended_v2_pythia_2p8b_m8_h25m_formal | 15 | 0 | 0 | 0 |
| planned_model_initial_timing_v1 | 2 | 0 | 0 | 0 |
| replay_fit_diagnostic_v2 | 15 | 0 | 0 | 0 |
| primary_independent_nll_test | 67 | 0 | 0 | 0 |
| supplement_A_independent_nll_test | 126 | 0 | 0 | 0 |
| supplement_B_independent_nll_test | 60 | 0 | 0 | 0 |
| C410_NLL_early_actual_v1 | 43 | 0 | 0 | 0 |

## 独立评价读取状态

主 NLL 首次读取时间：2026-11-06T07:14:59.369548+00:00；该授权只覆盖主 NLL，其他阶段和行为分数须各自满足解封条件。
补充 A NLL 首次允许语义读取时间：2026-11-06T09:47:43.509501+00:00；完整126条件及固定验证选择的18条件已解封。
补充 B NLL 首次允许语义读取时间：2026-11-06T11:49:02.974066+00:00；完整60条件及其初始基线已通过完整性门禁，全部结果已解封。
C：410M、100M目标暴露首次允许语义读取时间：2026-11-06T16:13:57.951572+00:00；整模型final/best结果已通过完整性门禁并完成分析。
C：1B、100M目标暴露首次允许语义读取时间：2026-11-06T23:57:20.638979+00:00；整模型final/best结果已通过完整性门禁并完成分析。
C：2.8B、25M目标暴露首次允许语义读取时间：2026-11-07T00:17:32.559156+00:00；整模型final/best结果已通过完整性门禁并完成分析。
C：410M行为评价首次允许语义读取时间：2026-11-06T23:15:41.880077+00:00；完整final行为端点已通过完整性门禁并完成分析。
C：1B行为评价首次允许语义读取时间：2026-11-06T23:58:27.886425+00:00；完整final行为端点已通过完整性门禁并完成分析。
C：2.8B行为评价首次允许语义读取时间：2026-11-07T00:30:55.025630+00:00；完整final行为端点已通过完整性门禁并完成分析。
各阶段分数仅在完整闭包与对应授权后解封；任务准备完成不代表分数已解封，报告结论以研究报告为准。

## 当前数据与选参记录

- `artifacts/data/VERSION`：v2
- `artifacts/evaluation_v3/VERSION`：v3
- [pilot_lr_selection_deterministic_v3.json](../results/pilot_lr_selection_deterministic_v3.json)：`{"selected_learning_rate": 3e-05, "selected_at_utc": "2026-11-06T04:33:28.433774+00:00"}`
- [pilot_lambda_selection_deterministic_v3.json](../results/pilot_lambda_selection_deterministic_v3.json)：`{"selected_shared_lambda": 0.05, "selected_mean_score": -0.031109066505816907, "selected_at_utc": "2026-11-06T04:42:13.601964+00:00"}`

## 入口

- [文献调研](../literature/文献调研.md)
- [研究路线与时间安排](研究路线与时间安排.md)
- [决策日志](决策日志.jsonl)
- 大模型与完整checkpoint：`/tmp/grad-research-20261106`。
- 本工作目录当前链接到临时盘；已有持久快照和主权重位于同级 `research_delivery/`。
- [持久主权重验收](primary_weights_durable_publication.json)；[快照记录](durable_snapshot_latest.json)。
