这是一份 CPU 恢复准备，尚未实际提取、解码或恢复 1B/2.8B 权重。固定例子为各模型原 Cfinal 的 **B4 seed11 final**，在读取其测试结果前指定；不是新增科学 seed，不纳入方法统计。两模型分别等待真实完整包、持久发布和 ROOT 验收，再由 ROOT 决定是否授权。当前所有模板均为 FALSE/pending。

准备目录：`artifacts/provenance/C_native_final_restore_preparation_v1/`。新入口复用包内已验证 decoder，不修改原 C410 恢复程序、live exporter、codec、publisher 或科学代码。

| 项目 | 1B | 2.8B |
|---|---|---|
| 固定 run | `extended_v2_pythia_1b_m32_formal_B4_s11_w0p05` | `extended_v2_pythia_2p8b_m8_h25m_formal_B4_s11_w0p05` |
| final 预测 token | 100M target / 5M replay | 25M target / 1.25M replay |
| 整阶段门禁 | 全 24 训练，24 logical / 21 unique，实际三对 D/B4 final 证明 | 全 15 训练 / 15 unique，无 D |
| durable TAR | `research_delivery/extended_C_1b_final_weights_lossless_v1.tar` | 同目录 `extended_C_2p8b_final_weights_lossless_v1.tar` |
| ROOT publication acceptance | `artifacts/provenance/C1B_final_publication_actual_v1/ROOT_acceptance.json` | `artifacts/provenance/C2p8B_final_publication_actual_v1/ROOT_acceptance.json` |
| 新隔离目录 | `/tmp/grad-research-20261106/reproduction_qa/native_Cfinal_1b_B4s11_final_v1` | 同根下 `native_Cfinal_2p8b_B4s11_final_v1` |

C410 已有例子需要 C410 final 包与 primary base 包两包合用。这里每个原 Cfinal TAR **自带对应模型 native 六文件 base/config/tokenizer/README**，恢复只需要该模型自己的单包、准备/授权小文件及已安装 Python/Torch/NumPy/safetensors。不要把新增 Cbest 包、旧 external-primary 方案或 145 member 常量套到这里。

未来 `bind` 只读明确列出的 JSON/源码与 TAR stat，不读 TAR header、ZIP 或 tensor。它要求实际 completion/manifest、原已审 native publication metadata reviewer、对应真实 whole-phase READY/identity、publisher 收据和 ROOT acceptance 全部 SHA 绑定。来源未齐、phase 未完成、FALSE acceptance、非 final、逻辑清单不全、任一训练非整数 rc0/有 stop/实际 argv 不同、源 checkpoint SHA 不一致等都会拒绝。完整来源/数据/config/summary/snapshot 和 1B D 证明的详细核查继续由已经审阅的 publication reviewer 提供，ROOT acceptance 明确绑定其实际报告和源码，不能用准备状态代替实际通过。

ROOT publication acceptance 的最小新增字段已列在各模型 `*_ROOT_publication_acceptance_FALSE_TEMPLATE.json`：实际发布后 `status="accepted"`、匹配 model/phase、`scope="original_Cfinal_native_package_publication"`，以及 completion、manifest、metadata_review、metadata_reviewer、publication 的精确 `{path,sha256}`。可以保留既有 acceptance 其他字段。这不是 CPU 恢复执行授权；后者必须另建文件，绑定最终 plan、runner/checker SHA、固定 run、scope 和 `sequential_only=true`，当前模板 `authorized=false`。

未来执行步骤如下；本次均未对真实包运行。1B 完整结束并审查后才单独考虑 2.8B，不能两模型同时运行。

1. 使用已有原 native 发布 runbook 完成 fullphase gate、metadata review、publisher、ROOT acceptance。保持源 TAR/收据；不要为了这次 QA 额外整包 `sha256sum`。实际源/目标全 TAR SHA 已由 exporter/publisher 完整实读支持。
2. 复制对应 `1b_sources_FALSE_TEMPLATE.json` 或 `2p8b_sources_FALSE_TEMPLATE.json` 为新的 actual sources 文件，填各**已经存在**小来源的 SHA，路径与真实 READY/identity 和 publication reviewer 保持一致。没有真实收据就停止，不能填假的 future SHA。metadata reviewer 固定 c596a56d…，其它 SHA 从实际小文件读取。
3. ROOT 保存来源清单 SHA，并运行以下 metadata-only bind（以 1B 为例，路径及 SHA 用实际值）：

```bash
CUDA_VISIBLE_DEVICES='' python artifacts/provenance/C_native_final_restore_preparation_v1/restore_native_final.py bind \
  --model pythia-1b --sources /ABS/NEW/1b_sources.json --sources-sha256 ACTUAL_SMALL_SOURCE_SPEC_SHA \
  --output /ABS/NEW/1b_bound_plan.json
```

4. ROOT 复核新 plan 和本准备源码清单，另存精确 execution authority；不能原地把本目录 FALSE 模板当成既有授权。仅在当前任务允许并符合资源/时间门禁后运行单例，保留下面 wrapper 的独立 evidence 目录：

```bash
CUDA_VISIBLE_DEVICES='' python artifacts/provenance/C_native_final_restore_preparation_v1/run_whole_child.py \
  --plan /ABS/NEW/1b_bound_plan.json --plan-sha256 ACTUAL_BOUND_PLAN_SHA \
  --authority /ABS/NEW/1b_ROOT_execution_authority.json --authority-sha256 ACTUAL_AUTHORITY_SHA \
  --evidence-dir /ABS/NEW/1b_restore_whole_child_evidence
```

新隔离目录必须不存在。新 runner 全局 nonblocking lock 只允许一个 native 恢复操作；ROOT 还要确认没有其它占资源的恢复 QA。子进程使用新的 cwd、`python -I`、清空 PYTHONPATH/PYTHONHOME 等，隐藏 CUDA、两 CPU 线程、HF offline 环境；归档 CLI 只从提取后的 `tools/src` 导入。检查程序自写完整 fingerprint，不导入 live 或归档 codec 的 fingerprint 实现。代码不发起网络调用；这里的 offline 环境不是 OS 网络 namespace 的隔离声明。

实际只提取 **12 个确定成员**：native 六文件、包 manifest、选定 B4 ZIP、自己的 metadata sidecar，以及归档 CLI/codec/包初始化三文件。TAR 先核 publisher 对应 dev/inode/size/uid/gid/mode/mtime/ctime，打开 fd 后再核，完成后再核；重复 header、路径逃逸、符号链接/硬链接成员、同名覆盖、size/SHA 不符均拒绝。对选定成员提取时 fresh SHA，保留实际 payload bytes 与 TAR header/选定内容交给 Python 的 serialized read bytes，后者不是物理磁盘 I/O。未选权重不提取，不重新散列整个 TAR。

归档 decoder 验 native base、逐 tensor raw SHA、完整模型 SHA 后写新 model-only checkpoint。独立 checker 对新恢复 tensor 按 sorted-key/dtype/shape/contiguous-uint8 算完整 FP，核包内原 FP、tensor 数量、decoder 完整 tensor hash 状态，核 ZIP 内原 CP SHA/kind、base/ZIP SHA、自己的 sidecar 身份/曝光；再 fresh SHA 新输出文件并对 decoder 报告。源 CP SHA 是已绑定的**来源证据**，这次不打开原 CP，也不重新与原 CP uint8 比较。没有 optimizer，不是 Adam resume、重训练或全模型阶段所有端点恢复验证。

资源和时间门禁（代理，不是 1B/2.8B 真实恢复实测保证）：

| 项目 | 1B 固定 draft metadata | 2.8B 固定 draft metadata |
|---|---:|---:|
| FP32 模型 raw bytes | 4,047,126,528 | 11,100,835,840 |
| 选定 ZIP bytes | 3,081,287,516 | 7,638,650,166 |
| scratch 下界约值 | 10.588 GiB | 24.749 GiB |
| 保守 reclaim-aware memory 门槛 | 19.077 GiB | 45.354 GiB |
| 执行等待 cap（另预留 teardown） | 600 s | 1,200 s |

scratch 实际 bind 使用选定 native/ZIP/sidecar/manifest + 新 FP32 模型 + 2GiB 序列化/工具余量；上表以当前固定 draft 的小 metadata 估计，实际包形成后重算，不能以本表冒充 future gate passed。RAM 使用 `4×FP32 raw+4GiB` 保守代理，执行前记录 cgroup max/current、anon/shmem/unevictable、active/inactive file、dirty/writeback 和 oom 事件。只计保守 clean inactive-file 可回收部分，不将 host free 或 `memory.current≈memory.max` 直接解释为匿名内存余量；不能把 max reclaim 计数当作 OOM。它不是资源预留或实际峰 RSS 保证。

保护结束点固定不晚于 **2026-11-07 02:56:32 UTC**，另留完整 30 分钟交付尾部到硬 deadline。执行 admission 要求剩余时间严格大于本模型执行等待 cap + 300s；即 1B 严格早于 02:41:32、2.8B 严格早于 02:31:32。300s 内明确覆盖 TERM grace 5s、KILL 后确认/回收 grace 5s 和 2s teardown bookkeeping 余量，wrapper 在 spawn 前重检。上限是带 timer 的机会窗口，不是执行承诺；ROOT 全科学训练/评价与其它已授权任务优先，余量不足就不做。

超时 wrapper 必须非零（124），即使 direct runner 收到 TERM 后正常退出 0。它对整个独立 PG 发 TERM，有限 grace 后只要整组仍存在就发 KILL；Linux child-subreaper 接管并回收孤儿后代，不能用 direct child 退出代替全 PG 消失。收据保存 group members 的 PID/PPID/state/PGID/session/start-time、信号、adopted zombie 回收和最终 `killpg(0)`/成员列表。若 D-state 或其它成员仍无法确认消失，结果明确为失败未确认（125），不能宣称保证已清理；ROOT 应处理实际残留。正常退出但留下后代同样清理并判失败。whole timer 包含终止/确认/回收，实际可以超过执行等待 cap，但 cleanup 预算已经位于保护余量内。

旧 13ea1a89… wrapper 的 toy 复现曾暴露“runner 退 0 而忽略 TERM 的后代存活”缺口，且未曾用于真实恢复。原源码、旧来源收据和测试证据保存在准备目录 `retained_before_timeout_repair_v1/`；当前修订不改变恢复/归档/科学实现，仅修尚未使用的新 wrapper 和拒绝/清理回归。

`whole_child_timing.json` 从真实 spawn 到 exit 记录 wall/user/sys/peak RSS，包含 runner、decode、independent checker；runner body 和各阶段均为嵌套子项，不能相加重复收费。metadata-only bind 若未来计时也是独立准备操作。只保存实际通过/失败，不把这次合成测试、历史 publisher hash 或 cache 当成新增真实恢复。
