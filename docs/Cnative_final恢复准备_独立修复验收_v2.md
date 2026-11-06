# C native final 恢复准备：独立修复验收 v2

**准备工程审阅通过。可以等待实际完整包、持久发布及 ROOT 验收后，再由 ROOT 按资源/时间门禁另签精确执行授权；本次没有授权或执行真实恢复。** 原 schema、12 成员闭包及指纹审阅见 [初审 v1](Cnative_final恢复准备_独立审阅_v1.md)。其超时缺口和原始失败探针完整保留；本页只补修后结论。

稳定 wrapper SHA：`2a2d09daac8504b099a5cd9682dff722bbad0519922ba39beedf4e96e9498266`。runner 仍为 `bfc748c776f18dff86c378c5c55ca72dbc63995cfc40ddfb96eb919486610db1`，checker 仍为 `6b7df26abef4ce0bd09dabbd7878a5a9585c8681d3cb97115f912dd542d3fa8d`。作者 v2 来源清单 36 项均 fresh 小文件核 SHA 一致，原 helper/core 保留。

## 独立复测

未重复原 29 项 suite。独立重放**同一原失败探针**，复制修后 wrapper 到临时目录，并使用原先的假 runner（父收 TERM 后退出 0；后代忽略 TERM）：

- wrapper 现为 **124**，`timed_out=true`，direct child 为 0；不会把父退出当作全组退出。
- 整 PG 经 TERM→5 秒 grace→KILL，subreaper 回收 adopted zombie；最后 `killpg(0)=false`、members=[]，独立检查后代 `/proc/PID` 不存在。
- wrapper whole wall **6.007773s**，包含 **5.007106s** 清理；独立探针父 wall **6.058141s**、user **0.175355s**、sys **0.136916s**。wrapper 是父探针的嵌套项，不能再相加。
- 模拟 persistent D-state 返回明确 `FAILED_*`、`whole_group_absence_confirmed=false`。静态确认 `run_group` 将未确认清理映射为 125，KILL 后没有无界 `wait()`。没有做真实 hung NFS/D-state 测试，也不声称 SIGKILL 能保证此类任务立即消失；若真实发生，ROOT 必须处理残留，相关 CPU 计时也可能未完整归属。

作者另外保留两种 leader 行为的真实 tiny PG 测试及来源：父 TERM 退 0、父也忽略 TERM；均 wrapper124、全 PG 确认消失。作者针对性 6 项测试计时与本次独立探针分别记录，不混算。

## cap 与交付尾部

600s/1200s 是**执行等待 cap**，whole 计时包括额外清理。TERM grace5s＋KILL 确认/回收5s＋bookkeeping allowance2s，共 12s；包含在固定 300s admission margin 内。spawn 前再次核剩余时间严格大于 `cap+max(300,12)`，保护端点仍为 2026-11-07 02:56:32 UTC：

| 模型 | 边界时刻（该时刻拒绝） | 提前 1 秒 | cap＋12s 预算最晚到达 |
|---|---|---|---|
| 1B | 02:41:32 | 通过 admission 的时间条件 | 02:51:44 |
| 2.8B | 02:31:32 | 通过 admission 的时间条件 | 02:51:44 |

两个严格边界与提前 1 秒均经纯函数独立核验；并非真实资源或来源 gate 已通过。这是预算/运营门禁，不是 kernel 或实时调度保证。实际执行必须使用修后 outer wrapper，ROOT 等它完成且确认无残留后才考虑下一模型；不能依据 direct runner 的退出或 body 收据提前释放顺序门禁。

最终来源与独立证据见 `artifacts/provenance/C_native_final_restore_independent_review_v2/COMPLETE.json`，其中绑定作者 v2 source/preparation/repair 收据、初审、原失败探针和修后独立探针。实际 completion/manifest、publication、ROOT acceptance、最新资源与时间条件尚需未来逐项满足；FALSE 模板保持 FALSE。原始大 TAR/ZIP/base/checkpoint、封存分数、GPU 均未读取或启动。
