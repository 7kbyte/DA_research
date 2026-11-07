# A-NLL 提前排程的纯时间评估

生成时间：2026-11-06T09:25:15.360720+00:00。仅供 root 决策；未授权、未追加或启动任何任务。

冻结 manifest：`/tmp/grad-research-20261106/worktree/artifacts/evaluation_ready_v5/supplement_A_nll/manifest.json`，SHA256 `88dc927c9b4909cca66901608216290e1e435022e4ff925c00875f5099077a90`。

真实完整网格126条件（108实际训练、18主运行别名），选择已冻结。新增 queue jobs 126：forward 108、reuse/materialization 18、initial 0。复用已完成的既有 initial 1 个和 canonical adapted origin 18 个，定义及完整输出 SHA 已核对。

训练时间口径取 observer 2026-11-06T09:24:19.181250+00:00，保留原10%周转余量。参考主 final 实测 whole-slot中位数 40.077859秒；SHA/scorer/polling均已包含，不重复相加。

|场景|forward/reuse秒|四卡理想增量min|独立A队列尾部秒|含10%+尾部增量min|23:50前剩余min|
|---|---:|---:|---:|---:|---:|
|low|40.078/10|18.79|15.08|20.91|38.75|
|central|50.097/20|24.04|19.90|26.78|32.88|
|high|60.117/40|30.05|19.77|33.39|26.27|

提前执行只将相同工作移到当前 master；最终评价的同额 slot 工作仅在这些任务真实完成且完整性通过后扣除。总推理工作减少为0。上述扣除目前为条件预算，不是假定任务已经完成。

独立A队列尾部按四卡均可用、原依赖已完成的FIFO模拟；现有非抢占children、长程C吞吐、共享CPU/I/O以及全master调度尾部仍有不确定性，不能当作按时完成保证。科学范围、选择规则和归档QA均不变。
