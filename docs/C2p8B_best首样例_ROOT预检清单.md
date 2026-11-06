# C2p8B best首样例：ROOT预检清单（非授权）

本清单只据小元数据、源码及stat整理；未读取CP/base/TAR/ZIP或评分，未运行exporter、创建TRUE许可或watcher。21:18 UTC观察时，本模型whole READY尚不存在，以下完整性门禁仍须实际就绪后重新核验。

**固定来源与范围。** 复用`artifacts/provenance/additional_endpoint_actual_v4/C410_best_sample_spec.json`的spec格式和原`run_one.py`，不改旧spec或旧授权。准确绑定：

- scope：`/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/scope.json`，SHA `7adb8f7f0461fd929995dd370ff2a59b2abc5c0d7750a39e745d21d5546d9de3`。
- exporter：`scripts/export_additional_endpoints.py`，SHA `c93e0c1d35fe39b3d71155a126db841b76648ba6b3afd58d475be4eb9c9da853`。
- wrapper：`artifacts/provenance/additional_endpoint_actual_v4/run_one.py`，SHA `030bf3fbc7d4b325adf4e30067a74650c7ff0d5c129204e6bc8e2aa9491c84ec`。
- 仅`--slot slot2 --package C2p8B_best --max-new-weights 1`；显式指定package，防止默认返回slot2首包B。CPU空CVD，OMP/MKL/OPENBLAS各2线程，cwd工作树，`PYTHONPATH=src:scripts`。ROOT未来另建精确spec和授权；新日志目录建议`artifacts/provenance/additional_endpoint_actual_v4/C2p8B_best_sample/`，必须尚不存在，避免旧timer碰撞。

**前驱与锁。** slot2前驱B_all_final已完成：completion为`/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/B_all_final/completion.json`（SHA `e49535b7fa156ca270d72d42559fedd73f3f330bb96b8b0cf56e357de452096c`）；完整父timer为`artifacts/provenance/additional_endpoint_actual_v4/B_all_final_batch/whole_child_timing.json`（SHA `8055f5fa406a03744844847ebbc8e8bb9867c536a530a62eca663fdd3d6087f8`，整数rc0）。对应`child_launch.json`记录child3272938/wrapper3272936，ROOT须再次确认两者退出。固定互斥锁是`/tmp/grad-research-20261106/delivery/.additional_endpoint_slot2.lock`；文件存在不代表持锁或空闲，预观察空闲不构成lease，exporter内部以非阻塞flock独占。最多两个新增编码worker，与原Cfinal合计最多三个，不绕锁换namespace。

**完整来源门禁。** 等`artifacts/evaluation_ready_v5/extended_pythia_2p8b_25m_nll/{READY.json,manifest.json}`真实发布并绑定实际SHA；核READY id/mode/status、manifest全部15个best与15个final身份。原15个训练记录必须恰为B0/B1/B2/B4/M×11/22/33：completed、严格整数rc0、无stop_requested_at、missing_outputs为空，实际argv/definition与冻结queue一致，summary/config/current小文件SHA与READY/snapshot交叉绑定。当前来源由scope中的`extended_v2_pythia_2p8b_m8_h25m_formal` context固定；合法master append不能替换任何原训练job对象。

既有`C_best_gate`再调用`complete_phase_gate`，对2.8B运行`audit_phase(...allow_incomplete=False,compare_weights=False)`，须passed/15 completed/无violations，输出`/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/C2p8B_best/fullphase_source_audit.json`。本模型没有D，`decisions={}`，不需要也不伪造D/B4 best等价证明，不预支alias；15 best均为独立端点。

注意：`audit_phase(compare_weights=False)`仍会fresh SHA读取`snapshot.model_files`（含2.8B base大载荷），不是metadata-only；ROOT首样预检不重复运行该audit，完整audit由exporter在编码前执行并计入同次whole-child parent，gate失败则保留失败、不续跑。

**固定首行及CP身份。** exporter按`(group=='D', seed, group)`排序，首样固定为`extended_v2_pythia_2p8b_m8_h25m_formal_B0_s11_w0`的`checkpoints/checkpoint_best.pt`，不按成绩选择。21:18:27的只读stat为dev139/inode6654340097/bytes11101003246/mtime_ns与ctime_ns均1793991550157645441；这是观察值，不能替代未来READY中的期望身份。配置当前SHA `d7c5898135d5d9c33164bf2266ceef4f1fcc57394bce5b79b3b2746eb15f4d5c`；`results/extended/pythia-2.8b/<run_id>_summary.json`当前SHA `56b96f7418e98040d4fa496aa996b1151ae84379d50c9fb68084d82431369b6e`，只做字节SHA，未解析评分。

ROOT未来重比canonical路径、非symlink、五stat及READY的CP SHA/size/mtime；exporter实际操作将核path/open-fd前后五stat、fresh CP SHA、实际best kind/selection/曝光和严格policy。首样metadata另写`C2p8B_best/metadata/<run_id>__best.source_metadata.pt`并记录SHA，record写`C2p8B_best/records/<run_id>__best.json`。事前不生成这些结果，不把final曝光替代best。

**容量、内存与时间。** reserve为`/data/202610/7kb/research_delivery/.extra_C_2p8b_best_weights_reserve_20261106`，当前dev106/inode331039199/uid=gid11471/mode0600/nlink1，size与allocated均150323855360B（140GiB），非symlink；未来需再次stat。实际完整增长证明为`artifacts/provenance/extra_endpoint_reservations_v1/C_2p8b_best_growth_receipt.json`，ROOT总容量验收为同目录`ROOT_full_capacity_acceptance.json`。此reserve不同于原Cfinal的140GiB预留，首样不会消费或发布它。

原scope没有2.8B首样内存准入数值，exporter只记录cgroup；不得沿用C410许可的8GiB。ROOT本次另定**fresh保守available≥48GiB**，参考既有22.76GB累计RSS代理并留余量，非峰值保证。以真实cgroup读取计算`clean=max(0,min(inactive_file,file-shmem)-dirty-writeback-unevictable)`、`available=max(0,max-current)+clean`；同时保留anon/shmem/unevictable、memory.events及scratch余量。不用host free或current≈max直接判断，低于门槛则暂缓，不清全局cache。

源码首样门槛是`now<2026-11-07T02:00:00Z`且`now+1200s+120s≤02:56:32Z`；**1200+120只是准入allowance，不是hard timeout**，原run_one没有超时终止功能。只允许首样，不授权batch或publication。样例后ROOT须重新审actual whole wall/user/sys/RSS、cgroup events、完整encode→decode→逐tensor uint8往返及metadata SHA，才能另行决定批量；整包剩余encode+assembly预计须严格早于02:00完成，publication及margin仍保护02:56:32。

原wrapper未来产生`child_launch.json`、`stdout.log`、`stderr.log`、`whole_child_timing.json`；whole child包含startup和endpoint，内部计时不得再加。单样record须`verified_additional_endpoint_not_yet_assembled`、一次编码及全tensor相等，`slot2_progress.json`应止于首样resource limit，不称15项整包完成。future实际运行由ROOT绑定新spec/authority后调用既有`run_one.py <spec>`；本页不执行此命令。
