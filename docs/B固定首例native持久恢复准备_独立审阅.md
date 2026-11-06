# B 固定首例 native 持久恢复准备：独立审阅

本次只读准备源码、小元数据和实际 durable TAR 的 `lstat`；没有打开 TAR header、ZIP、base、原 checkpoint 或评分文件，没有真实 bind/execute/恢复，也没有创建 ROOT true 许可。

固定例子是 `bankvar_B4_tr11_gen11_draw1001_k20_b0p05_w0p05` 的 final：training/generation seed 11、selection draw 1001、K20，306 steps，10,000,000 target / 500,000 replay prediction tokens。它与已发布 B manifest 唯一对应行完整一致，沿用原首个导出样例，不按分数挑选。预期 FP `05724eda…` 覆盖 405,334,016 个 FP32 元素、292 tensors；原 CP SHA `13dde87c…` 是来源身份，本次未重新读取原 CP。

独立 probe 核对计划的 12 个成员恰为本模型六个 native base/config/tokenizer/README 文件、完整包 manifest、该 run 的 XOR ZIP、自己的 metadata sidecar、三份归档工具。选中成员合计 **2,058,150,693 B**，SHA/大小分别交叉绑定已发布 manifest、原 v4 scope 和工具 pins；没有通过打开 TAR 来重新验证它们。未来真正提取时，沿归档工具读取 TAR headers 和选定成员，对这 12 项逐一 fresh SHA，不重复整包 SHA。当前 TAR 的 inode、容量、owner/mode、mtime/ctime 与 publisher 和 ROOT acceptance 一致；历史整包 SHA 不冒称本次重新计算。

归档 CLI 使用隔离新目录和 `python -I`，只导入提取后 `tools/src` 的 decoder。复用的独立 checker 以自己实现的 sorted-key/dtype/shape/contiguous-uint8 算法遍历新恢复完整 state，核对预期 FP、tensor 数量、FP32/CPU、source/ZIP/base provenance、decoder 全 tensor hash 状态、自己的 sidecar 与实际曝光，再 fresh hash 新恢复序列化文件。结果只覆盖这个 model-only 单例，不是 60 端点全恢复、与原 CP 再次 uint8 比较、Adam resume 或 GPU 重训。

初版新 runner `e3aa2b08…` 绕过了原 `native.execute()`，因此未取得原 C native 恢复的 `.native_Cfinal_restore.lock`。`sequential_only=true` 当时只是许可字段，不能机械阻止 B/C 同时恢复。作者已在新 runner `6d7245e78e225c3879eaf931b85e7be073b4ba9777d7263b7e0e1c39b1630a02` 修复：authorize 后，在建 QA、资源门禁和读取 TAR 前取得同一 nonblocking flock，持有至 body/收据结束；旧源码及旧计划原字节保留。独立临时锁 probe 验证锁在 body 内有效、异常后释放，且原 body 逻辑 AST 完整移入锁中，未改变恢复流程。

当前准备计划为 `bound_plan_v2.json`，SHA `61c19ee07d69936da329846fcb8aa8bd67ea8d8fe1ff8cd6b3084c032c3fa3c7`。独立比较确认它只更新 runner SHA 并增加显式锁路径，固定端点、12 成员、曝光、资源政策和预期 FP 均未变；新 FALSE 模板实际拒绝。原 native helpers/checker、归档 decoder 和科学代码未改。

复用的整 PG wrapper SHA `2a2d09da…` 与已独立验收版本一致：超时或残留后代必须非零，TERM → 有限 grace → KILL → subreaper 回收并检查整 PG；direct child 退出不能当作全组退出。若 D-state 等导致未确认清理，它记录显式失败，不承诺强制立即消失。这里未重新运行原大套测试或超时测试。

资源门槛为 scratch 8 GiB、cgroup reclaim-aware 可用代理 8 GiB，隐藏 CUDA、两 CPU 线程、offline 环境；不是峰值 RSS 保证。600 s 是 spawn 后 child/PG 的执行等待上限，另有最多约 12 s 清理预算，由 300 s margin 覆盖。以 02:56:32 UTC 为保护终点，准入须严格早于 02:41:32，并在实际 spawn 及读取 TAR 前重查。即使有余量，也须 ROOT 对最终 plan、runner、checker 精确 SHA 另行 true 授权，现有 FALSE 模板不能执行。

`outer_wrapper_parent_wall_seconds` 从 `main()` 内开始，覆盖 CLI/preflight、helper 加载、spawn 和整 PG 清理；不含解释器启动、顶层 imports 和最后收据序列化。`whole_child_process_wall_seconds` 是其内部 spawn-through-cleanup 子计时；body、提取、decode、checker 又在其内部。CPU 只能将该父计时的 self user/sys 和 children user/sys 各计一次，不能再相加各阶段。若需完整 Python 进程父墙钟，ROOT 须在外部另包 Popen/resource 计时，并把这些内部项保持为嵌套。

本次独立身份 probe whole child **0.100844 s**（user **0.071238 s** / sys **0.026033 s**），核对小 metadata、源代码及 TAR stat，并拒绝五种错误 scope。锁修订 probe whole child **0.056663 s**（user **0.049427 s** / sys **0.005937 s**），另核对同一 C 锁路径、实际 FALSE 拒绝、原 AST 和严格 900 s 拒绝／901 s 准入边界。两个 child 独立计时；人工阅读总父计时未知，不把子项之和称为完整审阅 elapsed。作者七项原纯 schema 检查和两项新锁测试只读收据，没有重复大套测试。

修订后的准备工具可等 ROOT 依据当时资源与剩余时间另行授权。稳定来源 SHA 以 [独立审阅目录](../artifacts/provenance/B_native_restore_independent_review_v1/) 的 `COMPLETE.json` 为准；这不是实际恢复成功收据，当前真实 QA/actual 目录仍应不存在。
