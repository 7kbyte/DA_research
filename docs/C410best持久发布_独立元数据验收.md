# C410 best 持久发布：独立元数据验收

依据完成的 publisher 小收据、progress/stdout、发布前核对、ROOT 精确许可/验收和当前 stat，支持该持久归档已完成。没有重新打开 TAR/header、ZIP、base、checkpoint、sidecar 或评分载荷，没有重跑发布或修改原源码。

目标为 `/data/202610/7kb/research_delivery/C410_best_weights_lossless_v1.tar`，**26,490,163,200 B**，原 publisher 已 fresh 验证的完整 SHA 为 `0e05b8b3d7f8088220b6a1cc9b8daf9d44b30eb758014d8732390361a22424a0`。本次只 stat：dev106 / inode467366543、11471:11471、0600、nlink1、大小和 mtime/ctime 均匹配完成收据。原 extra best reserve 名称已消失，目标保留其原 inode；scratch 源 TAR 仍在且 stat 稳定。

实际发布前核对器 `a6250f88…` 通过 **24 best 逻辑端点 / 21 独立编码 / 3 真正 D/B4 别名 / 155 TAR 成员**。ROOT 许可绑定该结果、原 manifest/completion、publisher 和 launch；保存的当时 master state 中，24 个完成记录逐项等于完整包记录，未要求仍会更新的整个 live state SHA 保持不变。包内含 native 六文件 base/config/tokenizer/README 和恢复工具，不依赖 primary 或 C final 包；model-only 不含 optimizer。本次不声称从 durable best TAR 做过实际独立恢复。

冻结 publisher `9c194743…` 的顺序是：fresh 整源 SHA → copy stream SHA → 文件 fsync → 整有效前缀 readback SHA → 截断已验证尾部并 fsync → 不覆盖的原 inode 提交 → parent directory fsync → passed 收据 → complete progress → 返回并输出 JSON。真实 receipt 为 passed，progress 为 passed/complete/published，stdout JSON 与 receipt 相同，publisher stderr 为空；durability events、SHA 和实际 stat 相互吻合。外层收据写入异常发生在此前持久工作完成之后。

**包装失败单列。** 外层脚本先将 exporter batch whole 副本复制到 `whole_child_timing.json`，publisher 完成后又以独占 `x` 写 publication whole，触发 FileExistsError，外层观察 exit1。旧内容已原字节改名 `batch_whole_child_timing.json`：其中 **1953.1042039617896 s**、CPU 和 rc0 均属于 exporter batch，且 `publication_performed=false`，不能再记作 publication。

publication 只有 **195.2169531546533 s 内部 region** 可精确引用：从 `publish()` 开始至 passed receipt 构造，已包含 durable commit，但排除随后的收据/progress 写入、FD 关闭及 stdout。完整 spawn-to-exit wall、user/sys CPU、peak RSS 和 publisher child returncode 未落盘，全部保留 unknown，不从 stdout、日历区间或 batch 副本补造 rc0/whole。异常、原 batch 副本、ROOT acceptance 和 durable adjacent receipt 均独立绑定。

稳定独立收据：[验收目录](../artifacts/provenance/C410_best_publication_independent_acceptance_v1/)。新的独立核对 child 成本另列；原 exporter、publisher 和 metadata reviewer 的父/子项不在此重复收费。
