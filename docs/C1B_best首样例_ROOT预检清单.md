# C1B best首样例：ROOT预检清单（非授权）

仅据小元数据、源码及stat准备；未读取CP/base/ZIP/TAR或评分，未创建TRUE/spec、运行exporter或测试。2026-11-06 22:03:20 UTC观察仍为23/24训练完成，`C_s33`运行中；whole completion、audit、两类proof及NLL READY尚未形成。以下门禁保持pending，不能据本页启动。

**固定执行范围。** scope为`/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/scope.json`，SHA `7adb8f7f0461fd929995dd370ff2a59b2abc5c0d7750a39e745d21d5546d9de3`；exporter `scripts/export_additional_endpoints.py` SHA `c93e0c1d35fe39b3d71155a126db841b76648ba6b3afd58d475be4eb9c9da853`；原wrapper `artifacts/provenance/additional_endpoint_actual_v4/run_one.py` SHA `030bf3fbc7d4b325adf4e30067a74650c7ff0d5c129204e6bc8e2aa9491c84ec`。仅显式`--slot slot1 --package C1B_best --max-new-weights 1`，不使用slot1默认首包。沿`C410_best_sample_spec.json`格式，由ROOT未来另绑精确spec/authority；空CUDA_VISIBLE_DEVICES，各CPU线程变量为2，cwd工作树，`PYTHONPATH=src:scripts`。未来日志目录用尚不存在的`artifacts/provenance/additional_endpoint_actual_v4/C1B_best_sample/`。

**前驱与互斥。** slot1前驱C410_best已完成并发布：

- completion：`/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/C410_best/completion.json`，SHA `1a2ccfa6238ba3636ad086d6ad7fc50c1a5f77b5e0a0a1731c8791455d04c20f`，24 logical/21 unique/3 aliases、文件及目录fsync完成。
- 完整batch父timer：`artifacts/provenance/additional_endpoint_actual_v4/C410_best_batch/whole_child_timing.json`，SHA `c24ba9a50917cf88478bdccd71c79feb2f0541fe4a2572afa155b68c99f74d5a`，严格整数rc0、whole1953.104204s；startup/endpoint/assembly均为嵌套。`child_launch.json` SHA `c3a662a1624a589132ef748befa96a9719c397f46437d86d431856d1f35be290`，child3346337/wrapper3346336当前均已退出，ROOT操作前再核。
- 发布：`docs/C410_best_weights_durable_publication.json` SHA `139e54cfcbae7763a1b651abd4b54c2646dbc1af99dff26787cce0b28107091a`；ROOT验收`artifacts/provenance/C410_best_publication_actual_v1/ROOT_acceptance.json` SHA `ce7c7905d5f7a424eb3832bd83080f17b906b5b45fddd685170338ef3623e3a8`。不要误用曾与batch同名碰撞的publication外层timer。

固定锁为`/tmp/grad-research-20261106/delivery/.additional_endpoint_slot1.lock`，exporter内取非阻塞排他flock。锁文件存在不表示空闲，事前观察也不构成lease；不改锁/namespace绕过互斥。C2p8B使用slot2，原Cfinal另持原有worker锁，本次不得触碰它们；新增slot1和slot2加原Cfinal最多三个编码worker。

**完整阶段门禁。** 必须等待`artifacts/evaluation_ready_v5/extended_pythia_1b_100m_nll/{READY.json,manifest.json}`真实发布，绑定其SHA。范围是八组B0/B1/B2/B3/B4/M/C/D×三seed11/22/33，共**24条训练状态、48个final/best端点身份**，不是30CP。逐训练核completed、`type(returncode) is int`且为0、无`stop_requested_at`、missing_outputs为空、实际argv/definition与冻结对象相同；config/summary当前小文件SHA、snapshot、READY输入及CP canonical路径/size/mtime交叉绑定。master合法追加不能替换原训练对象。

ROOT还须拒绝`stop_requested`及其他stop请求字段，不能只检查`stop_requested_at`。

同模型证明目录为`results/extended_phase_equivalence/extended_v2_pythia_1b_m32_formal/`，须实际有`completion.json`、`audit.json`、`final_model_equivalence_proofs.json`和`best_model_equivalence_proofs.json`。核24/24完整来源audit及全部source绑定；final与best各三对D/B4完整state指纹，明确seed/端点/CP SHA/实际selection与曝光。`C_best_gate`先核48身份中的24 best，再消费完整final gate和best proof；只有真实best equality为true的D端点才能alias，不预支21 unique或套用final的相等结论。即使不等也保留独立端点，不强制别名。

ROOT预检仅核这些小metadata/stat，不重复跑完整audit。exporter原有startup/input核验及编解码会读取实际载荷，不能称为metadata-only；其实际IO与时间计入该whole-child parent，门禁失败则保留失败，不续跑或绕过缓存/证明。

**固定首样。** 按`(group=='D', seed, group)`冻结排序，首行为`extended_v2_pythia_1b_m32_formal_B0_s11_w0`的`checkpoints/checkpoint_best.pt`。22:03观察为dev139/inode13118866992/bytes4047211586/mtime_ns=ctime_ns=1793976686699151466、非symlink；config SHA `425b616c52703bbc822f41437cdac7b905a23343745ac4b40244b5f7a4a54c32`，`results/extended/pythia-1b/<run_id>_summary.json` SHA `56af99273c832c9cee0be1b9d96b89230e206d62bca5664afe003a6557b000d9`。这些是历史观察，ROOT未来须按实际READY重核CP身份；exporter核同FD前后五stat、fresh CP SHA、best selection/曝光、metadata往返。不能用full100M代替best实际曝光。首样记录及sidecar分别写`C1B_best/records/<run_id>__best.json`、`C1B_best/metadata/<run_id>__best.source_metadata.pt`，当前不预造。

**资源与时间。** 75GiB reserve为`/data/202610/7kb/research_delivery/.extra_C_1b_best_weights_reserve_20261106`，当前dev106/inode375195308，uid=gid11471、0600、nlink1、非symlink，size与allocated均80530636800B。完整增长receipt为`artifacts/provenance/extra_endpoint_reservations_v1/C_1b_best_growth_receipt.json`，SHA `91791c13f04d477d1d7a7d7d47276c3ae240277aae615a07d40345dcefbf24ce`。ROOT操作前fresh stat；此reserve与原Cfinal的70GiB不同，首样不发布或消费它。

ROOT本次另定**fresh保守available≥24GiB**，这是独立新准入值，不是原scope已有条件或峰值保证，也不借用2.8B的48GiB。计算`clean=max(0,min(inactive_file,file-shmem)-dirty-writeback-unevictable)`、`available=max(0,max-current)+clean`，保留anon/shmem/unevictable、events及scratch余量；不足即暂缓，不以host free替代。

原首样时间门槛为`now<2026-11-07T02:00:00Z`且`now+420s+120s≤02:56:32Z`；**420+120只是准入allowance，不是hard timeout**，原run_one没有timeout终止功能。只允许首样；实际wall/user/sys/RSS、events、完整往返与metadata验收后，ROOT才能另定batch。批量需预计剩余encode+assembly全部严格早于02:00完成，publication及margin保护02:56:32。wrapper未来生成child_launch/stdout/stderr/whole_child_timing；whole含startup和endpoint，内部子计时不能重复相加。首样成功仅称一个已验收端点，不称24项整包完成或持久发布。
