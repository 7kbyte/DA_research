# B 固定首例 native 持久包实际恢复验收

这次实际完成的是一个事前固定的 CPU 恢复样例：B4，training seed 11、generation seed 11、bank draw 1001、K=20，final 第 306 步，目标/回放预测 token 曝光分别为 10,000,000 / 500,000。它来自已经持久发布的 `B_all_final_weights_lossless_v1.tar`，不按测试成绩选例。

ROOT 执行收据为严格整数 rc0，结束于 2026-11-06 18:50:34.942724 UTC。外层完整 spawn→exit wall 为 **36.184441 秒**，user/sys 为 22.344700 / 7.730223 秒，峰值 RSS 为 2,881,683,456 字节。内部进程组、runner body、提取、解码和独立指纹检查计时均已包含在这次外层 wall 内，不能再次相加。

执行时只从这一个 TAR 按需提取 12 个 member：包内 native base 的六个文件、固定样例 ZIP 与自己的 metadata、包 manifest 和三个归档 codec 源文件。提取收据中的 12 项完整 SHA/大小逐项匹配冻结 plan。选定 member 载荷共 2,058,150,693 字节，TAR 返回 Python 的字节共 2,058,355,626（含所经 header；不是物理磁盘 I/O 测量）。恢复在隔离目录、Python `-I`、清空 PYTHONPATH、离线配置和隐藏 GPU 下执行；解码使用包内 codec 与 native base，独立 checker 使用预先固定并复制进隔离目录的源码，不依赖原 checkpoint 或外部 live base。

归档 decoder 验证了全部 **292 个 tensor** 的原始 hash；独立 checker 重新计算完整模型指纹，与包内 encode/decode/原始 bytewise 验收记录及预期值一致：405,334,016 个元素，模型状态 SHA `05724eda110e91eb76c8f6d45b54386b374e486662207a266166a0fde0ffb790`。新恢复文件共 1,621,450,850 字节，其新序列化 SHA 为 `e3b026cfe90176462f56f734fb45f30ee6777f2f7bbb192a756958f66c60932d`；模型完整 tensor 载荷 hash 覆盖 1,621,336,064 字节。文件容器 SHA 与模型状态指纹分别核对，不把它们混为同一身份。

内部记录 `timed_out=false`、direct child rc0，PG3356096 在结束时及独立复核时均不存在，共享恢复锁未见持有记录。持久 TAR 当前 device/inode、大小、mtime/ctime、owner 和 mode 与发布/固定 plan 一致；新恢复文件 stat 也与 ROOT 验收一致。body admission、ROOT 验收及较后独立复核的 oom/oom_kill 均为 0；ROOT 验收时共享 cgroup `max` 增加 123，较后复核时累计增加 177。这是共享限额/回收压力计数，不能归因于这个 QA，也不能当作 OOM。

独立审阅绑定 [ROOT 验收](/data/202610/7kb/research/artifacts/provenance/B_native_restore_actual_parent_v1/ROOT_acceptance.json)（SHA `0194effe…`），核对已有小证据及文件 stat，**没有重新打开或散列 TAR、ZIP、base、新恢复模型或原 checkpoint**。九份实际 QA 小证据已由 ROOT 拷入项目，逐 SHA 相同。独立审阅的整体人工耗时未知；自动 metadata 检查子进程另有 0.129793 秒计时，不冒充整个审阅成本。

验收结论仅为：这一个预选 B final 模型能从其 native 持久包恢复，且已有执行证据中的完整状态指纹一致。它不代表重新恢复全部 60 个模型，不是本次 fresh 原 checkpoint uint8 比对、Adam resume、GPU 重训练或新评分。逐项核对见 [独立收据](/data/202610/7kb/research/artifacts/provenance/B_native_restore_independent_acceptance_v1/REVIEW.json)。
