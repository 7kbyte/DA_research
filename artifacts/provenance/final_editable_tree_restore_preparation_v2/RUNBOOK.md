# 最终可编辑工作树恢复：v2 原子完成标记准备版

当前只交付工具、FALSE 模板和临时小归档测试。最终 project archive 尚未产生；本目录没有执行真实提取、释放 8 GiB 预留或操作工作区 symlink。ROOT 必须在最终归档实际发布后单独批准执行。

ROOT 的真实 shared 探测确认目录 `renameat2(RENAME_NOREPLACE)` 返回 EINVAL，v1 在本平台不能完成目录发布，原字节保留。ROOT 随后确认同目录 regular-file 的 fsync→hardlink 无覆盖→移除原名→父目录 fsync 成功；两份实际探测见 [目录失败证据](../final_editable_tree_restore_preparation_v1/actual_durable_directory_commit_probe.json) 和 [文件 marker 成功证据](../final_editable_tree_restore_preparation_v1/actual_durable_file_marker_probe.json)。v2 使用原子 **marker**，不声称原子目录 rename，也不采用可覆盖 rename。

## 输入与新目录

薄工具 [restore_tree.py](restore_tree.py) 仅使用 Python 标准库。复用了既有发布收据的字段约定；既有权重恢复器只抽取选定成员，不能完成全树末尾 manifest 校验与预留容量释放，因此不直接用于本任务。

ROOT 复制 `plan.FALSE.template.json` 与 `authority.FALSE.template.json` 到新的操作证据目录，填写真实最终 archive、publication、ROOT acceptance、archive 旁 `.receipt.json` 的路径和 SHA。ROOT acceptance 与计划的 status 必须均为固定 `accepted_durable_final_snapshot`；1918 等中间快照仅作 schema 参考，不能冒充 final。不得把尚未产生的文件写成已验证。模板中的 pending 字段和 `authorized: false` 必须保留至实际审核；工具拒绝当前模板。

计划必须固定：archive 当前 `device/inode/bytes/mtime_ns/ctime_ns` 五项、压缩 SHA、payload 文件数、解包总 regular bytes 上限、独占新 namespace、原工作树路径，以及 11471:11471 所有权。archive、reserve 和新 namespace 必须处于同一个已存在、无 symlink 的 durable 父目录。恢复目标恒为 `<新namespace>/research`，证据为其旁 `restore_evidence/`，工具不操作原工作区路径。

计划与授权必须同时显式写 `publication_mode: verified_tree_with_atomic_completion_marker`。新 namespace 由独占 mkdir 创建；其中 `research/` 是从开始提取即存在的 **candidate**，其存在本身不能表示完成。

8 GiB reserve 的原始来源是 `artifacts/provenance/final_editable_tree_capacity_reserve_v1/receipt.json`。ROOT 还须绑定其当前完整 stat：dev/inode、uid/gid、0600、length、allocated blocks 和 mtime/ctime；原 receipt 记载的 dev/inode/owner/mode/length/blocks 必须保持一致，硬链接数须为 1。工具会独占 flock 并在 unlink 前再次检查。只能消费这一项 reserve；1+3 GiB 报告容量及其他权重 reserve 不在范围内。

运行授权必须逐项固定工具 SHA、计划 SHA、新 namespace/目标目录、源 archive 路径/SHA、四份来源收据 SHA、reserve 路径/stat 和 owner。`authorized` 必须是 JSON 布尔 `true` 且 `authorized_by` 为 `root`。模板不构成执行许可。

## ROOT 实际执行顺序

1. 等最终 archive 真正完成发布和 ROOT 验收，检查包旁收据与当前 stat。源 archive SHA 不能写进它自身的内容；本操作依靠包旁真实发布证据，不为内部 latest 指针的自引用反复重打包。
2. 审阅填写后的计划与实际空间，给出独立 ROOT 授权。保留填好文件的原始字节及 SHA，不能覆盖本目录 FALSE 模板。工具的 `max_total_regular_bytes` 包含末尾 manifest，须小于等于 reserve 的已分配 bytes；文件系统元数据、并发写入和空间回收时机仍可能导致 ENOSPC，8 GiB 预留不是跨文件的原子容量保证。当前树约 2.1 GiB、旧里程碑压缩包约 784 MB 仅为历史参考。

   实际 plan/authority、外层 stdout/stderr 和 whole timer 应保存在持久的新操作证据目录，或执行后将精确字节复制到新 namespace 的 `restore_evidence/`，不能只留 `/tmp`。publication/ROOT/adjacent 小收据可用精确 SHA 副本作绑定；由 ROOT 安排复制，工具不自动扩展范围。最终 archive 已冻结，无需将这些后建恢复证据反向写入原 archive。
3. 用一个完整外部 Popen/resource 计时入口记录 spawn→exit、严格 int returncode、stdout/stderr，并保持日志新文件无覆盖。可复用 ROOT 已有外层计时方式；平台没有 `/usr/bin/time`。不要再次为计时包装多层守护器。实际命令形状如下，替换为新证据目录中的真实路径：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python /ABS/PREPARATION/restore_tree.py \
  --plan /ABS/NEW_OPERATION/plan.json \
  --authority /ABS/NEW_OPERATION/ROOT_execution_authority.json \
  --execute
```

4. 必须同时满足：`restore_evidence/completion.json` 是完整且来源匹配的 `verified_tree_ready_marker`，外层 returncode 严格为 int 0，ROOT 实际验收通过。成功 stdout 状态为 `passed_verified_new_editable_tree_restored`，`completion_marker_durability` 为 `parent_fsynced`。仅目录或 marker 存在均不够。工具内部 wall 从 `restore()` 入口计起，包含其内嵌阶段，不能再加到外部 whole；不包含 Python 启动/参数处理。marker 内仅记录发布前 wall，不预先声称尚未发生的 post-link fsync；人工审阅耗时未知。
5. ROOT 等全部 writers/agents 结束后，另行决定是否切换工作区 symlink。工具既不切换，也不提供自动切换步骤。

## 工具的验收和失败边界

释放 reserve 前只读取有 SHA 绑定的小收据与 metadata/stat，依赖原 publisher 已完成的全包校验。源文件在打开后、释放前及解压后均核 path/fd 五项 stat。工具在自己的新 namespace 再探测一个小文件的 hardlink publication、目标同 inode，以及重复 link 必须 EEXIST；清理自己的两项探测名并 fsync。运行时若不支持，保留 reserve 并失败；不调用任何目录 rename。

然后 unlink/关闭仅自己的 reserve，并 fsync 父目录；同一个 gzip 提取流记录实际压缩 read bytes 和完整 SHA。每个 TAR 成员只允许 canonical `research/<相对路径>` regular file，拒绝绝对路径、`..`、重复、软/硬链接、设备、FIFO 和目录项；不调用 `extractall`。文件逐块写入和 SHA，末尾 `PACKAGE_MANIFEST.json` 必须覆盖每个 payload 的精确名称/长度/SHA，不能有遗漏 symlink；manifest 之后的隐藏第二个 TAR 成员也拒绝。

candidate 全部文件通过内容校验、逐文件 fsync、目录自底向上 fsync，并设为 11471:11471 后，才准备完成 marker。完整 JSON 先以新文件名 `restore_evidence/.completion.json.prepared` 写完、fsync、设定 owner/mode，再在同 evidence 目录 `os.link(temp, completion.json)` 无覆盖原子建立 marker，移除自己的 temp 名并 fsync evidence 父目录。marker 绑定 archive/五项 stat、plan/auth/tool/来源收据 SHA、exact candidate、manifest/count/逐成员 hash。其内容明确要求外层 rc0 及 ROOT 验收，不能自称已执行未来的 post-link fsync。成功 stdout 记录该 fsync 已完成、temp 已移除及 marker SHA。

文件仅保留 owner 的执行位，权限为 0600/0700，目录 0700；这恢复内容与可编辑性，不声称保留全部原 Unix 时间/权限 metadata。源 archive 和原树不删除、不移动。

验收记录包含 manifest SHA、逐成员实测 bytes/SHA、完整压缩流 SHA/bytes、源 stat 与发布/fsync 标志。提取内容与 archive 逐成员一致的声明截至提取验收；之后若 ROOT 增加最终导航/状态文件，须单独记录新增内容。最终发布/恢复权威证据保留在 archive 旁及新 namespace 的 `restore_evidence/`，无需也不能反向塞进原 archive。

任何失败都不触碰原树；新 namespace/candidate/日志保留，绝不假写完成或自动重试/清理。无 marker 时 candidate 明确未完成。若 marker 已出现后 fsync 或后续操作失败，failure 记录 `completion_marker_present: true`，并区分 `uncertain_until_parent_fsync`、`unknown_after_failed_publication` 或已知的 `parent_fsynced`；不能将其当作普通无 marker 的失败，也不能据 marker 自动切换。ROOT 必须保留证据人工判断，外层非 0 一律不接受本次恢复。

若 reserve 已释放，它不会自动恢复，空间不足时部分提取可能留下。极端 ENOSPC 若连 failure JSON 都无法 fsync，stderr 和 ROOT 外层日志是失败依据；`compressed_stream_bytes_read: null` 表示提取失败前未取得完整读量证据，成功提取后即使 marker 阶段失败，也保留已测完整读量。

本工具不提取项目外部权重 TAR，不恢复 Adam 状态、不运行 GPU、不等同于训练可重复性验收。原项目 archive 的 external-weight index 仍只是索引。
