本工具只准备未来真实 C410 READY 后的一次 CPU 时间预检。当前没有运行真实预检，也没有生成入队授权。入口为 `scripts/preflight_C410_ready_nll.py`；冻结 FIFO、scheduler、worker 和运行中的 v5 准备器均不修改。

未来由 ROOT 确认 C410 全部 24 项真实训练成功结束、v5 产生实际 READY 后，显式提供下列真实文件及 SHA，使用全新的输出目录：

```bash
python scripts/preflight_C410_ready_nll.py \
  --ready <C410_NLL_READY.json> --ready-sha256 <SHA> \
  --manifest <实际manifest.json> --manifest-sha256 <SHA> \
  --phase-queue <实际queue.json> --phase-queue-sha256 <SHA> \
  --master-queue-sha256 <当时live_master_queue的SHA> \
  --output <新的只读预检证据目录>
```

默认读取原 master PID 2988506、`results/runtime_budget_latest.json`、`logs/supplemental_job_state.json` 和 `configs/supplemental_v1_queue.json`。这条示例没有执行。工具不调用 scorer、不解析分数、不读 checkpoint payload，不写活队列、状态、排序或 READY，不启动 GPU、watcher 或 scheduler。

预检要求 READY 绑定实际 manifest/queue/spec，全 24 项状态为 completed、rc=0、无 stop、输出存在，完整 audit 与 final/best 三 seed 的等价证明已形成。manifest 必须覆盖 24×2 个逻辑端点；物理 forward 数量直接从真实 canonical jobs 取得，不预设 43 或 49。不等价端点保留单独推理，任何别名必须绑定真实完整 model-state 证明。matched-extended initial 必须新增，adapted jobs 均依赖它，不能借用主 v3 的 NLL baseline。未定义 rate 的外部 reuse 等新情形会拒绝，交 ROOT 另审。

同时检查：runtime 来源 SHA、90 秒新鲜度、全部 63 项实际状态与原训练定义、当时批准的 live master 顺序、原 master 的 queue/state/deadline/GPU argv、四个实际 occupied slots 及其子进程 argv。保留小 JSON、源码、proc argv 的原始字节；每份输入打开前后核 stat 和实际 SHA，计算结束再次核 stat/字节。任何状态、进程、定义或源文件竞态均拒绝，不自动放宽或回写；下一次只能用新的真实快照重试。

比较项仅为当前 C-only 与真实 C410 NLL priority 插入。两者都使用实际四个 running slots，不抢占；保留全部未完成 C 训练及原 pending 顺序。模拟从当时捕获时刻起算，READY 已实际形成的等待不追溯为零，也不把旧 runtime 之后的进度提前抵扣。新 matched initial 必须等实际 occupied slot 自然释放。v5 high 的 initial/final/best rate 已含 1.5；不再乘第二次。冻结 FIFO 对全部剩余 duration 仅乘一次 1.1。

完整保存全部 63 项实际或预测 start。未完成训练的 latest launch 必须不晚于 22:25，whole-master finish 必须不晚于 23:35，分别保留原严格 22:30 准入前 300 秒、23:50 停止前 900 秒的额外余量。恰好 22:25/23:35 可通过，一微秒越界拒绝。通过时间门槛仍不是入队授权，ROOT 必须另作实际 scope、源和新鲜度复核；这也不是运行时间保证。

完整 v5 最终评价 high tail 仍原封保留作为重复 scope 的保守 envelope；其中 C410 NLL 在早评模拟与尾部各出现一次，不能据此计算总费用。预检不扣尾部；只有实际整个同定义 C410 NLL 完成后，未来预算才可扣一次。该粗 envelope 不替代最终评价真实队列的逐 job 准入/FIFO 检查，完整 30 分钟交付尾余量继续保留。

本轮测试只用临时目录和手工 CPU duration，覆盖门槛边界、canonical 合并计价、四槽自然占位、dependency、一次 10% 和快照变更拒绝；没有运行任何真实阶段预览。
