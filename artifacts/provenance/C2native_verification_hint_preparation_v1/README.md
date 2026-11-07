# C2 native CPU 恢复前的四个 derived verification 输出提示：仅准备

本目录没有执行hint。固定四项为已完成 C2p8B_best 的 B1_s22、B0_s22、M_s11、B4_s11 restored.pt，合计44,404,036,408 B（≤48 GiB），均388tensor完整decode+原bytewise通过。不是原训练CP、base、ZIP或TAR；不选当前未完成输出。历史整文件SHA/完整FP、record SHA、当前dev/ino/size/mtime/ctime及owner/nlink已绑定；准备时全系统FD/maps无引用。该检查只是当时观察，执行必须重做，不是writer lease。四个合格输出足够，未选择TAR fallback。

`prepared_plan.json`不是执行授权；FALSE模板不会执行。ROOT须审阅本script与plan，再独立生成精确authority，字段保持模板结构，仅明确写入authorized:true（不得编辑FALSE原模板）。新authority同时绑定plan与script完整SHA。运行`hint_once.py`必须显式`--execute`和两份SHA；现未创建TRUE。

执行前至少三次按真实cgroup公式重新检查：入口、metadata/proc检查后、只读shared flock后。任何一次保守available≥48,698,310,656 B，正常写SKIP并不调用fsync/DONTNEED；前两次SKIP甚至不打开target FD。门槛是本次native恢复4×raw+4GiB，不借用或降低此前encoder48GiB门槛。原失败gate d55f6…14,605,336,576 B原字节保留。准备时新观察21,532,065,792 B不构成未来准入。

仍不足时，仅4个固定derived输出以O_RDONLY|O_NOFOLLOW打开；逐文件shared flock；同FD/path完整stat再核；只放行自己的4个新readonly FD，任何其他FD/maps/不可读proc均拒绝；同FD fsync后仅offset0到exact已验证长度DONTNEED，前后stat不变、record SHA仍同。所有FD finally关闭。advice_attempts/completed_advice_count逐项记录，失败/部分操作不得冒充无操作或完整成功。shared锁为advisory，未占用/修改live exporter全局锁。

本操作不读payload、不删除/改字节、不做global drop_caches；fadvise是建议，不保证回收多少。前后cgroup变化不能归因出exact reclaimed bytes。无论提示结果如何，都不构成C2恢复authority；ROOT后续仍需原门槛fresh admission。

未来命令必须在所有依赖步骤已成功后单独执行，shell使用`set -euo pipefail`，新receipt与计时路径须不存在。例如ROOT将精确authority写至新路径后：

```bash
set -euo pipefail
# DIR、AUTH、AUTH_SHA、ACTUAL_DIR 均由ROOT明确填写；ACTUAL_DIR独立且已存在。
/usr/bin/time -v -o "$ACTUAL_DIR/whole_child_time.txt" \
  /usr/bin/python "$DIR/hint_once.py" --execute \
  --plan "$DIR/prepared_plan.json" \
  --plan-sha256 badb933059a381fce7fea70dc07cb548e853f5c058ccab401ccb77f95e40ff32 \
  --authority "$AUTH" --authority-sha256 "$AUTH_SHA" \
  --receipt "$ACTUAL_DIR/receipt.json"
```

ROOT也可沿既有Popen父timer方式记录strict rc、spawn→exit wall/user/sys；script只报告body region，不假称整个进程计时。whole父项与内部fsync/advice region不可相加。准备子进程whole.07006311044096947s/user.050971/sys.017584，手工工程父项未知；未运行任何测试或实际hint。
