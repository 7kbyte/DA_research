# 研究运行状态

更新时间：2026-11-06T04:00:02.782423+00:00
截止时间：2026-11-07 03:26:32 UTC（北京时间11:26）
已用 0.56 小时；剩余 23.44 小时。

以下状态来自运行日志。配置完成不等于实验完成；正式结论以最终研究报告为准。

| 调度阶段 | 完成 | 运行 | 待执行 | 失败/阻塞 |
|---|---:|---:|---:|---:|
| lambda_job_state | 16 | 0 | 0 | 0 |
| lambda_v3_job_state | 6 | 3 | 7 | 0 |
| v3_lr_eval_job_state | 4 | 0 | 0 | 0 |

## 当前数据与选参记录

- `artifacts/data/VERSION`：v2
- `artifacts/evaluation_v3/VERSION`：v3
- [pilot_lr_selection_v3.json](../results/pilot_lr_selection_v3.json)：`{"selected_learning_rate": 3e-05, "selected_at_utc": "2026-11-06T03:56:50.711816+00:00"}`

## 入口

- [文献调研](../literature/文献调研.md)
- [研究路线与时间安排](研究路线与时间安排.md)
- [决策日志](决策日志.jsonl)
- 大模型与完整checkpoint：`/tmp/grad-research-20261106`。
- 数据、代码、结果、日志与文档保存在本工作目录；最终将注明关键权重的持久化位置。
