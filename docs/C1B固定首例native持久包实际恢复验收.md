# 1B 固定首例 native 持久包实际恢复验收

独立小元数据核对通过。恢复对象事前固定为 B4 seed11 final：3058 steps、100,000,000 target 与5,000,000 replay prediction tokens。实际执行从已发布的原 Cfinal 单 TAR 提取12个选定 member，包括配套 native base/config/tokenizer/README 六文件、该 run ZIP/metadata与归档恢复工具，未使用 live base、原训练 CP、网络或 GPU。

成员提取时逐项完整 SHA 验收；decoder核196个 tensor，另一个 CPU checker重新计算196 tensor、1,011,781,632 elements的完整模型 fingerprint，均为 `da1fe7db068d1111f973920ac59fb6c0ad70914c288cfdef434db9d7bef42d82`。新恢复文件完整SHA为 `b9480d726b7d126c194ca8f5e23c921951a40abf696e883e3161f38c0992233d`，序列化4,047,222,722 B。来源 CP SHA 被精确绑定；这不是重新打开原 CP 的 fresh uint8 比较，也不验证 Adam/optimizer resume 或重新训练。

实际12成员 payload为5,175,105,157 B，TAR返回Python5,175,337,756 B（含header）；这是序列化读取量而非物理磁盘I/O。checker完整tensor指纹区域为4,047,126,528 B，另对新序列化文件全读SHA。历史整TAR SHA来自已验收publisher；此次恢复没有重算整TAR，本次独立审阅也未读取TAR/ZIP/CP/base/恢复模型载荷。

实际 whole child严格整数rc=0，84.33324014395475s，user49.866894/sys20.015561s，RSS6,555,627,520 B。body84.271704s、extraction27.711255s、decode/checker与清理计时都是嵌套子项，不能再加到whole。没有timeout或清理信号；PG3460499结束时与独立审阅时均完全不存在，共用恢复锁当前无持有者，该观察不是锁租约。cgroup oom/oom_kill保持0；共享cgroup max计数变化不能归因本次QA。

这证明一个固定模型端点的离线model-only恢复，不代表全部24端点都实际恢复，也不增加科学seed或支持性能结论。独立审阅whole.067276459s另计，手工审阅/文档父耗时未知。未创建ROOT acceptance、操作2.8B恢复或缓存。

证据：[独立收据](../artifacts/provenance/C1native_CPU_restore_actual_independent_v1/receipt.json)，其中绑定ROOT精确authority、当前持久TAR stat、source SHA与全部实际小证据；`small_actual_evidence`保留可打包的小文件副本。
