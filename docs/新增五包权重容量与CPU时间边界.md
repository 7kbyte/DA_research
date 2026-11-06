五包容量在当前记录下可容纳；时间限制项是 **C2.8B best**。本规划没有启动 reservation、export、GPU、watcher 或新授权，没有改现有排序或 helper。实际容量 writer 已由 ROOT 独立启动，本规划仅只读其固定 sequence 和文件 stat。

115+75+30+75+140=**435 GiB**，即 467,077,693,440 字节。按 ROOT 先前报告的 free，全部分配后余 397.294 GiB；实际 sequence 记录的 initial 分配前 free 为 887,097,413,632 字节，对应全部目标后余约 **391.174 GiB**，高于 16 GiB floor。五个 initial 1 GiB 已有实际 inode；15:05 的 stat 只看到首包继续增长，不能冒充五包完整容量。当前 free 不再减完整 435 GiB，以免把 initial/正在增长部分扣两次。满额 grow 收据、容量验收和资源首例门禁仍须 ROOT 核准；既有 reservation 不动。这些 ceiling 是实际包必须遵守的上限，尚无 best 压缩大小的数学保证；超 cap 应拒绝该可选包，不挪用原任务空间。

固定两条独立 CPU 链：slot1 为 A93→C410best→C1Bbest，slot2 为 B60→C2.8best；每个 2 threads，与原 Cfinal 最多三个编码 worker。新五包 publication 在估算中串行，原 Cfinal、科学训练/评价及交付始终优先。并行三个 worker 的磁盘竞争没有独立实测，25% 是敏感性场景，不是保证。

ROOT 补充的 15:05 cgroup 实测为 88 CPU quota、memory.max 约 1.031 TB，current 约 1.026 TB，主要为 file cache；oom/oom_kill 均为 0。不能把宿主可用 2 TB 或 os.cpu_count=192 当本进程可用容量，也不能将 memory.events 的 max/reclaim 计数称为 OOM。原首例 exporter 的累计 RSS 高水位约 22.755 GB；两个新增 worker 可暂以额外约 45.51 GB、三者约 68 GB 作资源代理，实际要经 clean cache 回收，并非已获保证的空闲内存。各新 slot 必须先过真实样例 RSS、耗时及 memory.events 检查；本规划不据缓存可回收这一点自行启动。

| 包 | 独立编码数：无别名 / 条件成立后 | full-roundtrip proxy 分钟：无别名 / 条件后 | 另留 assembly / publication 分钟 |
|---|---:|---:|---:|
| A remaining final | 93 / 93 | 157.861 / 157.861 | 9.035 / 15.958 |
| B final | 60 / 60 | 101.846 / 101.846 | 5.915 / 10.408 |
| C410 best | 24 / 21 | 39.949 / 34.956 | 2.406 / 4.163 |
| C1B best | 24 / 21 | 100.169 / 87.648 | 6.001 / 10.408 |
| C2.8B best | 15 / 15 | 178.122 / 178.122 | 11.331 / 19.427 |

A/B 用已完成 A15 的完整父 wall 1527.687871 秒÷15；C 用已核过原 final 的单份完整父 wall：410M 99.872976、1B 250.422683、2.8B 712.486536 秒，包括 encode/decode/逐 tensor 验收。它们不是 best 实测速率，尤其 1B/2.8B 仅各一个原 final 样本。条件 21/21 只能在三 seed 各自 best 的完整状态等价证明实际成立后采用；没有证明仍为 24/24。C2.8B 无 D 对照，保留 15，不预支其他别名。

assembly 使用原两包同起点累计 timer 的差分（before-TAR→包完成），不把累计前缀相加；采用较慢约 **229.472 MB/s**，乘 package ceiling 并另加一次 native-base-size pass。publication 采用三个实际收据中较慢约 **128.962 MB/s**，按完整 ceiling 计，不因别名预先缩小。新包 native base/config/tokenizer/README 的字节仅 stat，没有读其 payload。未实测 proof/READY 等待仍未知，不能设为零。A15 父 timer 已含旧 assembly，本表另留 assembly 是有意的保守 forecast 重叠，不可当作新增实际成本重复登记。

规划唯一沿用 12:47 已保存 `last4_1b_after_2p8b` 两个高情景，不读取 order-agnostic aggregate 推模型完成时间：

| 已存训练高情景 | C410 完整训练 | C1B 完整训练 | C2.8B 完整训练 |
|---|---|---|---|
| C-only | 15:33:00 | 23:04:44 | 22:07:14 |
| 假想 C410 早 NLL | 15:33:00 | 23:20:28 | 22:22:58 |

这些不是实际 READY 时刻，未含额外 audit/proof/准备延迟。工具保留零延迟的乐观下界和**另加 30 分钟的敏感性**，不会凭它们生成 READY。训练预测已含一次 10%，不再乘第二次；新 CPU 全链另外比较原 proxy 与 ×1.25。

对限制项 C2.8B，×1.25 的结果如下，15 份编码数量在两个别名情景相同：

| 已存训练情景 / 额外 READY 延迟 | 最后一次新操作开始 | 整包 assembly+publication 完成 | 两个门槛 |
|---|---|---|---|
| C-only / 0（仅乐观下界） | 01:35:03 | 02:28:21 | 数学 proxy 内通过 |
| C-only / 30 分钟 | 02:05:03 | 02:58:21 | 两项均不通过 |
| 假想早 NLL / 0（仅乐观下界） | 01:50:47 | 02:44:05 | 仅余约 9.21 分新开工缓冲 |
| 假想早 NLL / 30 分钟 | 02:20:47 | 03:14:05 | 两项均不通过 |

新操作必须严格早于 **02:00** 开始；已经开始的一次操作可自然结束，但不意味整包可拖过 **02:56:32**（硬 deadline 前完整 30 分钟交付尾部）。在上述 ×1.25 proxy、没有额外 publication 排队/原 Cfinal 抢占的孤立边界下，C2.8B 完整实际 READY 最晚须早于约 **22:32:11**；相对两条已存训练预测，未知 READY 延迟最多约 **24.94 / 9.21 分钟**。这是工程粗界，实际资源变慢、原任务优先或发布等待会进一步提前，不能把这段余量当已保证。C1B 的对应孤立开工界约 00:00（24 份）或 00:15:39（已证 21 份）；C410 余量较大。

因此 A/B/C410 边界较宽，可等资源样例验收和实际满额容量后由 ROOT 决定；C1B/C2.8B best 只能在真实全阶段 READY 到达时，以当时剩余完整包、已发生的 IO 和新速率重新检查。若无法在 02:00 前启动最后一份、或无法在保护尾部前完成 assembly+publication，**开始该可选 phase 前即拒绝**，保全原科学范围、原 Cfinal 和交付；不靠事后中断或只发布便利子集。30 分钟额外 READY 敏感性已使 C2.8B 不通过，现阶段不能把五包全部交付当确定承诺。

来源字节和全部 16 个固定情景保存在 `results/optional_weight_expansion_planning_v1/plan.json` 及 `sources/`，producer 在 `artifacts/provenance/optional_weight_expansion_planning_v1/produce.py`。没有重新搜索顺序、读取封存分数或大 payload。本规划起点 15:05:08 也只是两新 slot 假想可用时刻，并非实际 launch；后续决定必须沿用 ROOT 的实际容量/资源/完整 READY 门禁。
