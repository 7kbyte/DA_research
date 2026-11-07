# C五组：实际successor进度与ROOT运行清单

只读观察state.updated_at_utc：2026-11-06T23:09:04.191052+00:00。这不是原check，不核评分/attestation载荷，也不是授权或冻结分析state。

| 阶段 | closure | strict0已完成 | running | pending | 定义/已启动argv | 无stop字段 |
|---|---:|---:|---:|---:|---|---|
| extended_pythia_1b_100m_nll | 43 | 0 | 0 | 43 | True | True |
| extended_pythia_2p8b_25m_nll | 31 | 0 | 0 | 31 | True | True |
| extended_pythia_410m_100m_behavior | 22 | 16 | 4 | 2 | True | True |
| extended_pythia_1b_100m_behavior | 22 | 1 | 0 | 21 | True | True |
| extended_pythia_2p8b_25m_behavior | 16 | 1 | 0 | 15 | True | True |

全部closure job ID及逐项元数据在closure_job_metadata.csv；精确原CLI、plan/repair SHA、authority字段与未来输出路径在future_commands_PREPARED_ONLY.json。未改变既有计划、原protocol、cache或固定first-read路径。

1. 某一组全部closure completed、严格整数rc0、无任何stop字段、definition/实际argv一致且无missing输出后，ROOT在该组全新的operation目录独占保存实际successor完整state原字节副本，计算并固定其SHA。其他phase可以继续运行；不得使用旧master或本次观察摘要替代原state。
2. 保持该副本path/SHA不变，ROOT用对应check_argv运行原v3 check，并完整记录外部父wall/user/sys、rc、stdout/stderr。check将执行原完整source/output/attestation和CP缓存current-stat/允许fresh fallback门禁；本清单没有预先通过这些门禁。检查ready=true，scheduler_state及其SHA精确等于绑定的副本，并绑定check输出SHA。
3. ROOT新建独立授权文件：从保留的FALSE模板逐字段继承plan/protocol/READY/manifest/expected_counts/checkpoint_verification，authorized_by必须小写root、allow_partial=false、冻结选择字段不变；显式加入本组operational_repair、成功check证明及actual_successor_immutable_state path/SHA。只有ROOT明确许可后才设置allow_score_analysis=true并计算新文件SHA。C410行为的旧FALSE没有repair/state字段，须在新的ROOT文件中增加，原模板不可覆盖。
4. 使用相同state副本、真实新authority SHA及全新analysis output，运行对应原analyze_argv；保持5000次bootstrap、seed20261106及原端点/统计循环。原analyze会再次完整inspect，并在语义解析前独占创建固定first-read journal。不得预写、覆盖或重置journal，失败必须保留并停止，不能换输出目录绕过首次读取。
5. check与analyze分别保留完整父计时；内部guard/verification计时是嵌套项，不能重复加总。只有原analyze实际rc0且本组COMPLETE及其来源闭合后，才进入独立数值复核/报告。

本次只创建进度表和命令参考；没有读取任何评分/CP/base/归档载荷，没有运行原check/analyze、创建TRUE或首次读取journal，也未启动GPU、修改queue/core/plan。当前state的SHA仅绑定本次观察，未来必须由ROOT重新保存实际完整state原字节。
