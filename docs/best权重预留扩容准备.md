# best 权重预留扩容器：仅准备，尚未启动

工具 [grow_reserved_file.py](../scripts/grow_reserved_file.py) 仅扩展 `/data/202610/7kb/research_delivery/.primary_best_weights_reserve_20261106`。绑定计划 [primary_best_storage_growth_plan.json](primary_best_storage_growth_plan.json) 的 SHA256 为 `087fc67fd932f87c6a17cb394a463625ae01dac07d3e8fa351125938e9b71f83`。准备时实测 device=106、inode=594318300、uid/gid=11471、权限0600，长度及实际分配均为 10,737,418,240 字节；目标为 best TAR 的 37,979,074,560 字节。本次只生成代码、计划与临时文件测试，没有启动共享盘写入。

打开预留文件时使用 `O_NOFOLLOW|O_APPEND`，不创建文件、不调用 truncate。获得非阻塞独占 flock 后，直到最后一次文件 fsync 完成才关闭 fd、释放锁。每次追加前核对 fd 与路径的 inode/device/owner/权限/长度，路径被替换、外部改长度或遇到其他错误即停。追加内容全部为零，每次检查最多 256MiB，每轮最多 2GiB，轮间等待30秒。按实际文件系统剩余容量扣除2GiB，再按分配粒度向下取整；其他进程仍可能同时消耗容量，因此 ENOSPC 另有保护。

容量不足时保留原文件并等待；短写或 fsync 的 ENOSPC 记录实际 partial size/blocks 并重试，不回收已写尾部。每段及每轮 fsync 后刷新本地进度。TERM/INT、2026-11-07 02:00 UTC 或达到目标均触发同步退出；非 ENOSPC 错误停止并保留实际文件。正常关闭后 receipt 明确 `lock_released` 与 `final_file_fsync_completed`。root 必须确认进程已退出，再使用已验收 publisher；本工具自身不会发布 TAR。它不访问 GPU，不触碰1GiB/3GiB报告预留或其他共享文件。

CPU 临时文件测试共21项，另连同 publisher 8项共29项通过：增量/旧前缀与 inode 保留、容量不足零写入、短写 ENOSPC 留 partial、ENOSPC恢复、延迟 fsync ENOSPC恢复、五种身份绑定、初始长度、路径替换、双进程锁、signal/deadline边界、真实子进程 TERM 后同步解锁、计划SHA拒绝、非 ENOSPC 停止及 symlink 拒绝。测试文件为 [test_reserved_file_growth.py](../tests/test_reserved_file_growth.py)。

root 审阅后的启动命令如下，本次未执行：

```bash
nohup env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python scripts/grow_reserved_file.py \
  --plan docs/primary_best_storage_growth_plan.json \
  --expected-plan-sha256 087fc67fd932f87c6a17cb394a463625ae01dac07d3e8fa351125938e9b71f83 \
  --progress logs/primary_best_storage_growth.progress.json \
  --receipt docs/primary_best_storage_growth_receipt.json \
  > logs/primary_best_storage_growth.log 2>&1 < /dev/null &
```

若进程停止时尚未达到目标，保留该进度与收据。重新启动必须由新计划明确绑定当前实际长度及同一 inode，旧计划不应覆盖或静默接受变化后的长度。扩容完成仅代表容量具备，best TAR 在 publisher 完成全源/前缀 SHA 验证与发布收据之前仍只在 scratch。
