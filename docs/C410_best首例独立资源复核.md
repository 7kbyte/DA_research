# C410 best 固定首例独立资源复核

结论：小证据与资源样例通过复核，**不是剩余批次或 publication 授权**。没有重新打开 CP/ZIP/TAR/base，也没有复做 encode/decode；核验的是已发生样例的完整证明、源 SHA 和当前 stat。可追溯收据见 [REVIEW.json](../artifacts/provenance/C410_best_sample_independent_review_v1/REVIEW.json)。

首例按冻结顺序取 B0 seed11 best，不依表现挑选。原 CP SHA `dd25db838ede6d137cd630818e5ecb3ad8f9100ef47eba86740a48f04a260bf4` 与 ROOT 样例 authority、实际 record、C410 best manifest 一致；实际 step1224、target40,029,696、replay0，所属完整训练为100M。config/summary fresh小文件SHA、原 definition/argv 与 completed整数rc0/no-stop 证明一致；源 path/open-FD 前后五项 stat 与当前 stat 相符。

样例 encode/decode/逐tensor uint8 比较三处完整 state FP 一致：`d9be7b307cf922d7399cd4aee405c1ca2368c19b45c32423684ff76a4b08b9fd`，405,334,016元素、292 tensors；metadata往返通过。native base SHA `3e09001631bb18ec8a6f4522f8e6ab26a53099e69228f9822adc91bfbe27511b` 与冻结 scope/encode 相符；本次仅取 base/restored/sidecar 当前 stat，不将历史payload SHA说成本次重新读取。恢复是 model-only，无 Adam resume。

C410 whole24 rc0 与独立 phase audit/completion/READY均已存在；manifest含24final+24best。三seed D/B4 **best** 的两侧源CP SHA、config/summary SHA、实际曝光和完整FP与该manifest绑定，三组均equal。故剩余计划为20个非alias编码目标，加3个具有资格的D metadata/源SHA核验；alias尚须由实际 exporter按证明完成，不能称已导出。21个编码目标也不预断言21种不同模型FP。

样例 whole child wall **105.168669581s**、user **95.059167s**、sys **11.064906s**、RSS **3,686,518,784B（3.433GiB）**。其中 endpoint parent92.790311620s，内含 encode69.780135788/decode18.203947183/uint8verify4.494729701s；这些都是嵌套，不相加。样例尚未 assembly/publication，当前不是可交付完整包。

共享 cgroup 样例前后 `oom/oom_kill/oom_group_kill=0`；`max`增加58，表明触及共享限制/回收路径，不能解释成无内存压力。18:24:39观察仍无OOM，clean-inactive信用下保守可用约41.094GB，scratch约23.697TB；直接headroom很小，依赖可回收文件页，这不是内存预约或未来峰值保证。ROOT应在真正批授权/启动时重新核共享cgroup、原Cfinal与两slot并发；本次不获取锁、不批准额外worker。

按 ROOT 指定保守参数：20×120s + 三alias核验总360s + assembly540s = **3300s（55min）**到预计 publication 开始；900s publication +600s margin 后总 **4800s（80min）**。alias360与assembly540已共同构成 `assembly_allowance_seconds=900`，不得额外再加900。120s超过样例endpoint parent且超过105.169s whole，但仍是规划上界假设，非所有端点实测。

以18:24:39.881249观察为起点，publication预计19:19:39.881249开始，含publication/margin预计19:44:39.881249结束，满足严格publication开始<02:00及完成≤02:56:32。两门槛推出：publication约束要求启动**严格早于01:05:00 UTC**；finish约束单独允许≤01:36:32，前者约束更紧。未来实际启动须重算，不把这份早期观察当运行许可。

另查 v7 operational ledger：v6 collector whole **6.833468527s**未记入；v7成功collector whole **8.851966519s**也未递归记入自身账。两者均仅列下一版候选，不能再加它们内部collector body；v7原source/产物未改。首例本身也发生在v7来源截止后，应待后续明确登记。
