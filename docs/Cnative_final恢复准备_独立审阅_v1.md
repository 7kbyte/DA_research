# C native final 恢复准备：独立审阅 v1

本次只读源码、小 JSON 和既有 29 项测试收据；未调用真实 bind/execute，未打开真实 TAR/ZIP/base/checkpoint，未创建真实授权或启动 GPU。另运行一次 1 秒 cap 的假 runner 进程组探针，已清理全部探针后代。证据目录：`artifacts/provenance/C_native_final_restore_independent_review_v1/`。

**初审结论：原 native schema、12 成员及独立指纹可对接，但新 wrapper 的超时收尾存在实质缺口。作者正在修复；修后复核和真实包验收前不应授权执行。** 本说明不取代后续修复验收，也不改原准备/归档代码。

| 核查项 | 只读结论 |
|---|---|
| exporter→completion/manifest→metadata review→publisher→ROOT acceptance | `build_plan` 字段与原 native exporter 相符。原 publication reviewer c596a56d…已细核 frozen config/summary/snapshot、final 曝光、严格 rc0/no-stop/argv 和 1B 实际 D/B4 证明；新的 acceptance 模板要求绑定各精确 path/SHA，不将准备状态当实际发布通过。 |
| 固定范围 | 原 B4 seed11 final：1B 100M/5M，2.8B 25M/1.25M。要求全阶段实际完成及 24 logical/21 unique/3 alias 或 15/15/0。1B 三对未证相等时不可强迫此计数；仅固定单例恢复，不是新 seed 或全组恢复验收。 |
| 确切闭包 | native 六文件＋包 manifest＋固定 B4 ZIP＋自己 sidecar＋归档 CLI/codec/__init__ 三文件，共 12。原 exporter 的成员名与三项工具 pins 一致。恢复使用包内 base/tools，无 live codec/base 回退。 |
| 安全提取 | 固定对应 native 目录；selected name 规范化、duplicate header、regular selected member、size/SHA、exclusive output 和 TAR path/fd 前后身份限制适当。未选成员不提取；这不等于逐个检查所有未选成员类型。TAR 历史全 SHA 与本次 selected-member fresh SHA 分开。 |
| 指纹/精度 | decoder 逐 tensor XOR 恢复 FP32，核 native SHA、FP32 转换 raw SHA、全部 tensor/full-state hash；checker 自写 sorted-key/dtype/shape/uint8 摘要并核元素/tensor 数、ZIP/CP/base 来源、sidecar 身份、曝光和新文件 fresh SHA。归档 CLI 通过自己的 `tools/src` 导入。该操作是 CPU FP32 权重恢复，不是 BF16 推理。来源 CP SHA 为历史证据，本次不读原 CP；model-only 无 Adam。 |
| 时间/资源 | CPU2/CVD空、全局单例 lock、固定新 namespace、disk/reclaim-aware cgroup 代理、600/1200s cap＋300s admission margin、02:56:32 保护端点。必须用 outer wrapper；单独 execute 不为 extraction 提供完整 wall cap。资源是代理而非 RAM 预留/峰值保证。 |
| 成本 | 外层 spawn→exit 是父项，runner/extract/decode/checker 为嵌套子项，不重复相加。bind、既有 29 tests、人工准备另列；serialized Python reads 不是物理磁盘 I/O。 |

## 已证实并交作者修复的问题

原 `run_whole_child.py:46–52` 超时后向整 PG 发 TERM，却只等待 runner。若 runner 先退出，仍活的后代不再触发 KILL 分支。第 61 行又直接 `SystemExit(rc)`，runner 收 TERM 退出 0 时，wrapper 会在 `timed_out=true` 下返回 0。

独立探针仅复制 wrapper 到临时目录，放入完全独立的假 runner：父 TERM 后退出 0，子忽略 TERM。1 秒 cap 后实测 **wrapper rc0、timed_out=true、后代仍活**。探针随后显式 KILL 自己的 PG，确认后代 `absent`。未调用真实恢复 runner、有效授权或权重包。探针完整 wall **1.056895s**、user **0.047350s**、sys **0.017894s**。

修后验收要求：timeout 必须非零；grace 后无论 direct child 是否退出都检查并清理剩余 PG，whole 计时包含 teardown；cap＋grace/余量必须与 02:56:32 一致。若 kernel D-state 导致不能确认清理，须如实记录，不能声称已保证全组退出。无需真实 NFS hung 测试。ROOT 已要求作者只修这份未执行的新运营工具并保留旧源码/收据，原冻结科学代码不动。
