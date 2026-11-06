# 1B / 2.8B 完整组 NLL 独立复核准备

本版只准备独立 reviewer。真实模型的 test NLL、逐块输出和行为结果未在准备阶段读取，也未调用 check/analyze。旧 `review_C_nll_results.py`、旧测试和 C410 已用结果保留原字节。

`scripts/review_C_model_nll_results_v2.py` 接受两种固定范围：

|模型|真实训练|final + best 逻辑端点|预定对照|
|---|---:|---:|---|
|pythia-1b|八组 × 三 seed = 24|48|M−B4、C−B4、B2−B1、B4−B3、B3−B1、B4−B2|
|pythia-2.8b|B0/B1/B2/B4/M × 三 seed = 15|30|M−B4、B2−B1、B4−B2|

1B 的完整目标曝光为 100M，2.8B 为 25M。best 使用实际验证集选点的曝光，不补齐至 final。物理推理数从真实 manifest 与逐项完整模型状态证明计算，不预设有六个 alias；没有 alias 时仍输出具有完整表头的空 CSV。每个实际训练只在成本中计一次，D 与 B4 即使状态相同也保留两次真实训练费用。

执行前须已有 ROOT 合法解封并完成的整模型 `COMPLETE.json`。reviewer 逐项绑定 authority、plan、manifest、operational repair、原 first-read journal、readiness 和实际 READY。它不会生成权限、修复凭据或 first-read journal，也不会运行原 guard/collector/统计程序。checkpoint、基础权重、tensor、TAR、ZIP 的读取入口被拒绝；完整模型等价证明只读其已有小 JSON 与源码 SHA。

ROOT 应在该模型全部 inference closure 成功后，保存**实际 successor executor state 的原字节不可变快照**，让原 check/analyze 使用该快照。reviewer 的 `--state` 路径和 SHA 必须同时等于 COMPLETE readiness 所绑定的路径和 SHA。`--queue` 是 ROOT 明确提供并绑定 SHA 的实际 successor 不可变队列；只核本模型完整 baseline/endpoint closure，不解析其他 phase 的分数。每个 closure job 都须精确匹配冻结 definition、实际 wrapper argv、integer rc0、completed、无任何 stop_requested 字段、无 missing outputs。不会要求仍运行其他 phase 的 live state 在 bootstrap 期间不变。

数值部分保留旧版 documents、delta_vectors、raw_metrics、Resampling、stats、favorable、validate_grid 的 AST。独立从 chunk 汇总文档，核 token 加权与等文档指标、三 seed mean/sample SD、所有原有六指标的固定 5000 次 / seed 20261106 条件 bootstrap。没有新增估计量、对照、筛选或显著性结论。输出原规范的全部 CSV 与四张三行两列图（PNG/PDF/SVG），图中曝光说明从实际端点读取。

未来调用示例（本轮未执行真实数据；尖括号必须替换为 ROOT 当时实际文件及已核 SHA）：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
python scripts/review_C_model_nll_results_v2.py \
  --model pythia-1b \
  --complete <1B已解封结果目录/COMPLETE.json> --complete-sha256 <COMPLETE_SHA> \
  --authority-sha256 <该COMPLETE绑定的ROOT_AUTHORITY_SHA> \
  --queue <实际successor不可变queue.json> --queue-sha256 <QUEUE_SHA> \
  --state <ROOT实际executor不可变state_snapshot.json> --state-sha256 <STATE_SHA> \
  --check-timing <该模型实际check_whole_child_timing.json> --check-timing-sha256 <CHECK_TIMER_SHA> \
  --analyze-timing <该模型实际analyze_whole_child_timing.json> --analyze-timing-sha256 <ANALYZE_TIMER_SHA> \
  --output results/C1B_nll_independent_review_v2_actual1
```

2.8B 使用同一命令，将 `--model` 换为 `pythia-2.8b`，并将 COMPLETE、authority、所有状态/队列/计时凭据和输出目录替换为**该模型本次真实来源**，例如输出 `results/C2p8B_nll_independent_review_v2_actual1`。不得套用 C410 或 1B 的 timer、主 v3 初始 NLL、旧 master state。

check/analyze timer 必须描述成功完整子进程并绑定同一 plan、repair、实际 snapshot state；analyze 还须绑定同一 authority 和 COMPLETE 所在输出目录。scheduler whole-slot 包含 worker 核验与 scorer，teacher forward 是 scorer 内子项；这些成本分别列出且不重复相加。实际 reviewer 自身的完整 spawn→exit wall/user/sys 应由 ROOT 外层计时，内部 receipt 的 inside-run wall 只是子项。

准备证据位于 `artifacts/provenance/C_model_nll_review_v2_preparation/`，其中包括旧版本副本、手工 toy、各次定向测试日志、完整子进程计时、两模型 toy CLI 全部输出和最终源码 SHA 清单。toy 命名和记录明确为 SYNTHETIC，不是实际模型结果或真实解封权限。
