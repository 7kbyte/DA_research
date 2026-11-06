# 最终评价handoff：607项master的只读差异复核

本次观察于2026-11-06 17:13 UTC；精确输入副本、来源SHA、job ID及definition digest见[收据](../artifacts/provenance/final_handoff_readonly_review_v2/review_receipt.json)。只读取队列、状态、READY、manifest、缓存及源码元数据；未调用handoff／评分／分析，也未读或散列评分输出、checkpoint。原交接工具和历史文档不改，已有测试不重复运行。

## 当前仍不能交接

master有607项，607个definition digest均与state一致；578 completed且严格整数rc0、无stop，4 running、25 pending。PID2988506仍活着，四个历史child process group存活，state的active_jobs有四项，且确实持有`logs/supplemental_job_state.lock`。实际argv**没有**`--exit-when-idle`；不能把将来的idle误认为已经退出，也不能现在发停止信号。

补充计划实际5/9 READY：A-NLL、A-behavior、B-NLL、C410-NLL、C410-behavior。C1B和C2.8B各NLL/behavior四项仍waiting_training；`results/extended_run_audit.json`目前不存在。不得把旧文档“5个来源／6个待READY”的历史数字当当前状态；现在是主2＋补充5的7个实际evaluation sources，最终仍须主2＋补充9=11个。

| 当前实际evaluation source | queue项 | 与607 master完全同定义的项 |
| --- | ---: | ---: |
| main NLL | 67 | 67 |
| main behavior | 67 | 1 |
| A NLL | 126 | 126 |
| A behavior | 18 | 0 |
| B NLL | 60 | 60 |
| C410 NLL | 43 | 43 |
| C410 behavior | 21 | 0 |
| 当前所需依赖闭包 | **402** | **297** |

三个explicit prior sources仍是main NLL、main behavior、planned-model behavior initial。把三份prior队列也全纳入原始job并集为**404**；其中1B／2.8B的两个planned initial当前尚无已READY dependent，所以最小依赖闭包只含402，不能凭prior声明就另加两次推理。当前闭包中105项未在master：main66 behavior、A18（15 forward＋3 reuse）、C41021 behavior。297项只是结构上的继承候选；本次未重新核输出/attestation，不能宣称已完成真实handoff继承。其余310个旧master ID不在当前闭包，包含训练、诊断及暂未需要的两个planned initial，不能整份复制旧state充当评价队列。

最终四个C来源尚无实际队列，当前不能给最终精确物理job并集。预算v6的“余224=221 forwards＋3 materialization”是未预支1B未来等价alias的既有保守scope，不能当已经生成的224项queue，也不能再扣一次已完成main/A/B/C410。未来真实READY可能依法减少等价forward，须以实际manifest和双方CP绑定的proof为准。

## 缓存与新分析plan的边界

本次通过可变`cache_progress.json.cache_snapshot`取得并fresh核对了不可变snapshot：`digest_cache_snapshot_1793985172249449033.json`，SHA `933541044cd98ce82508d9e48b9854e7a1ddbb8eab0bcef60b4aab1ad4a951fe`；其progress时间17:12:52，仍有58个prewarm pending。这只是**本次观察绑定**，不是最终执行时的最新选择。

最终spec必须另存`digest_cache:{path,sha256}`，从届时最新progress取得snapshot并核reader、receipt、expected SHA及当前五项stat。不能用目录、mutable index或当前这个尚不完整的snapshot冒充最终覆盖；缺项将fresh fallback，应提前量出剩余bytes和所需时间。prepare/start各有自己的验证器，真实worker仍fresh核输入。cache hit是历史完整实读凭证复用，不是此次又读checkpoint。

ROOT新准备的A／C410行为**分析plan**及其固定cache是独立的首次解封门禁，不等于final handoff spec、不改变source queue定义、不授权现在推理或读取行为结果；本次没有重做prepare。

## ROOT实际切换清单（待条件真实满足，非本次执行）

1. 取得最终607队列与state一致快照：**每项**completed、`type(returncode) is int`且为0、无`stop_requested`／`stop_requested_at`、无缺输出，argv/definition仍同冻结定义，active_jobs=[]。63项C综合audit真实passed，三模型分别24/24/15完整；九个实际READY及其源/queue/manifest SHA均齐备，不能从partial summary补造。
2. 取新v5 draft，确认无pending_plan_ids，主2＋补9实际sources和三份显式prior组成的唯一job并集、完整initial ancestry、同ID字典完全相等。只将冻结worker单GPU任务纳入，不混训练或诊断。用届时最新immutable cache绑定新finalized spec；保留example草稿和所有旧证据。**不要现在移除example_only或造TRUE authority。**
3. 用真实最终FIFO及各job冻结expected_seconds核准入和完成尾部；包含实际audit/READY、hash补读、handoff、统计与发布费用，未测时间不能填0。预算v6在当时C-only训练预测22:58:38后，high完成01:41:31；若00:00才开始，其high完成02:42:53，仅距02:56:32交付尾部边界13.65min。它是LPT代理、含300s未测handoff allowance，不是最终真实FIFO或当前新预测；最后必须刷新运行状态并逐job核准入，不再机械叠一次10%或把whole-slot子时长双加。
4. 仅ROOT在以上条件全部满足且旧master真正idle时，才可对**已核身份的旧scheduler**作graceful终止。先确认无活跃科学任务，随后等待scheduler PID完全消失/reap、所有历史child PID/PG都退出，复核旧lock释放及四卡无遗留任务。不改margin、不重启旧训练master。仅旧master idle不够。
5. ROOT审核并冻结新spec SHA后，使用现成helper，不另写workflow：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python scripts/handoff_evaluation.py prepare \
  --spec '<新FINAL_SPEC>' --spec-sha256 '<审定SHA>' \
  --output-dir '<全新handoff目录>'

# ROOT审阅queue/state/old snapshots、实际继承凭证和预算后才执行：
python scripts/handoff_evaluation.py start \
  --handoff '<新目录/handoff.json>' --handoff-sha256 '<审定SHA>'
```

prepare会取得旧lock并核旧进程完全退出；新queue/state独立，旧文件逐字节留存。只继承定义一致且canonical／alias输出attestation和fresh输出SHA完整的项，reuse还核原始canonical。缺凭证恢复pending，不能把原summary当新推理完成。start再核来源、旧快照与继承证明，并把旧lock传给唯一后继scheduler；不得同时另开GPU调度器。

## 固定边界及helper自身没有代做的检查

hard deadline仍为11月7日03:26:32 UTC。tail1800、grace90、poll10，故stop margin1910、launch margin2210：**02:54:42**开始对running children发SIGTERM，90s后仍活着则SIGKILL；正常poll/退出后保护02:56:32起完整30min交付尾部。不是允许running一直到hard deadline。

每job严格要求`实际start + expected_seconds < 02:49:42`，等号拒绝。因此冻结allowance180／300／600s的start分别须严格早于**02:46:42／02:44:42／02:39:42**。不得因实测更快改allowance。helper start本身到stop窗口才拒绝，并不替代这些更早的逐job准入检查；一个能启动的scheduler也可能无任何task再获准入。

后继已显式带`--exit-when-idle`。但它的idle退出条件允许failed/blocked等非pending状态，停止窗口也可带pending退出；所以**scheduler rc0不等于全评价成功**。最终仍逐项完整性和每phase解封门禁验收。

`handoff_evaluation.py`只核所选来源、旧master退出/锁和相关任务继承，不替ROOT强制“全607成功／全63audit／全9READY／active_jobs=[]／四卡空闲”。其继承条件源码用`returncode==0`而非严格整数类型，且只拒`stop_requested_at`；因此上面第1步更严格的外部门禁必须保留。本次实际578个completed记录均通过该严格检查，未发现真假rc0混淆；这不是修改冻结helper的建议。

以上是准备差异说明，不授权终止master、GPU推理、TRUE authority或任何封存分数读取。观测child整wall0.194558s／user0.114920s／sys0.063330s，人工源码阅读/写作未全程仪表化，未冒充0成本。
