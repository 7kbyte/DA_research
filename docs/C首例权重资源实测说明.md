# C 首例权重资源实测独立复核

复核状态：complete。仅工程记录与文件身份审查，不读取 heldout NLL 或行为评分，不改变运行源码、原始记录或参数。

| 模型 | 首个完整记录 | FP32 原字节 | ZIP 字节 | ZIP/raw | 往返 wall 秒 | 往返 CPU 秒 | 进程累计 RSS GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| pythia-410m | extended_v2_pythia_410m_m64_formal_B2_s11_w0p05 | 1,621,336,064 | 1,229,118,349 | 0.7581 | 99.87 | 100.70 | 3.434 |
| pythia-1b | extended_v2_pythia_1b_m32_formal_B4_s11_w0p05 | 4,047,126,528 | 3,081,287,516 | 0.7614 | 250.42 | 251.39 | 21.193 |
| pythia-2.8b | extended_v2_pythia_2p8b_m8_h25m_formal_B4_s11_w0p05 | 11,100,835,840 | 7,638,650,166 | 0.6881 | 712.49 | 713.15 | 21.193 |

| 模型 | encode wall/CPU 秒 | decode wall/CPU 秒 | 原 CP 全 uint8 验收 wall/CPU 秒 | tensor 数 |
|---|---:|---:|---:|---:|
| pythia-410m | 74.92/75.29 | 17.12/17.70 | 7.48/7.60 | 292 |
| pythia-2.8b | 509.72/510.88 | 147.70/147.78 | 53.53/54.44 | 388 |
| pythia-1b | 187.77/188.20 | 42.86/43.60 | 19.15/19.48 | 196 |

每模型单例不是压缩上界或未来速度保证。ZIP/raw 的分母为全部 FP32 model_state 原始字节，不是包含 optimizer 的完整训练 checkpoint。编码、解码、逐字节验收计时是完整往返记录的嵌套子项，不与父项再次相加。父项另含 source metadata、部分 SHA/文件操作等开销。

CPU 秒是进程使用的处理器时间，wall 秒是实际经过时间；两者不相同。这里的单模型 timer 从源 metadata/编码开始，不包含此前等待 A 的 worker 锁；daemon 自启动 wall 还可能包含锁等待、轮询和无可导出 run 的等待，不能按其差额虚报精确等待时长。顶层 progress 在模型工作期间未刷新也不能据此判断卡住。

RSS 字段是整个 exporter 进程截至该记录的累计高水位，不能解释为该模型独占峰值、增量或同时运行内存总和。worker=1、torch threads=2、GPU=false；单例结果不能泛化到不同系统负载。

原生产者对每个模型已执行 encode→decode→原 checkpoint 全 tensor uint8 比较，记录 all_tensor_bytes_equal=true、tensor 数、模型指纹及完整 source/ZIP SHA；本次独立复核核它们的一致关联、FP32 字节数和每 tensor 原始/基底 SHA 元数据，并实际比较原 CP 与 sidecar 的全部非权重/非 optimizer metadata、mmap tensor 描述及当前文件 stat。本次没有重复扫描全部原 CP/ZIP/恢复权重内容，不能称第二次独立全字节验收。原文件均保留。

这些记录仍是 verified_draft_not_deliverable。完整 phase、READY 身份、独立审计、逐对 D/B4 final 证明通过以前，不能称完整模型包或持久化交付；D 是否 alias 由完整 final 证明决定。

A 发布复核：passed_current_identity_stat_review。若已完成，使用 publisher 的全源、复制及读回 SHA 验收证据，另独立复核当前 owner/inode/dev/permissions/size/mtime/ctime；本次没有重复 hash 整个 TAR。所有源记录及收据 SHA 详见同名 JSON。
