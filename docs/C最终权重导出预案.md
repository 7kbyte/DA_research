# C 最终权重 CPU 草稿与完整交付预案

当前状态：仅实现、临时目录测试及静态预检，尚未启动 C 真实权重编码、推理或共享盘写入。A 独立 exporter 已由 root 启动，其源码、codec 和既有 helper 未修改。本适配器不决定任何科学参数，也不依据验证或测试成绩选择保存端点。

保存范围由 `results/extended_formal_freeze.json` 固定：410M 24 final、1B 24 final、2.8B 15 final，共 63 个逻辑端点；100M/100M/25M 曝光保持各模型原协议。这里不包含 C best、B60，也不将不同模型/数据/horizon 混为一个比较。D 的实际唯一模型数以逐对 final 证明为准，不能预先假设 57 unique。

执行分为两层。每个完成训练 run 可先生成明确标为 `verified_draft_not_deliverable` 的 scratch 草稿。必须同时满足：scheduler completed/rc0、该训练 job 的固定定义及 digest 未变、实际 argv 与固定 argv 完全一致且无 stop_requested_at 字段、配置和输入 snapshot SHA 未变、完整 source checkpoint metadata 与 summary 一致、实际 execution policy 与本模型 matched extended initial 一致、final 曝光和每 step 累积预算正确、验证轨迹完整、无 error event。每个新模型都经过原 codec encode→decode→原 checkpoint 全部 uint8 逐 tensor 比较，并保存所有非权重/非 optimizer metadata 的精确往返 sidecar。初次编码实读的整个 source checkpoint SHA 与后续固定 C NLL **身份 manifest** 再次交叉核验；该 manifest 只取身份，不读取任何输出评分。

完整包只有在模型内全 24/24/15 条完成、完整源审计通过、final 身份 manifest 经 READY.json 正式发布后生成。READY 必须明确 ready=true/status=ready、id 和 nll mode 正确，并以固定 manifest 路径和当前完整 bytes SHA 绑定身份；没有 READY 时不读取正在构建的 manifest。410M/1B 另外核独立 watcher 的 completion、audit、final equivalence proof 以及它们绑定的来源 SHA；completion 的 passed 本身不能授权 D alias。每个 seed 必须有 exact final pair：同模型/phase/revision/seed/原 CP SHA，且 `bitwise_model_state_equal` 为真并与两边指纹一致时，D 才引用其 B4 ZIP。若为假，D 单独 encode/decode/全字节验收。D 保留自己的 metadata、原 CP SHA、曝光和完整证明。2.8B 无 D，整 cohort 经过既有独立 `audit_phase`。任何不完整或身份冲突都不能封包。

master queue 是合法可追加、可重排的调度入口。适配器每轮核当前各训练 job 的原冻结定义及 scheduler digest，不长期锁定整个可变 master 文件 SHA；原 phase queue、registry、config、snapshot 仍严格固定。

CPU 实际导出共用 `/tmp/grad-research-20261106/delivery/.supplemental_weight_export.lock`，A 持有时 C 等待，只有一个编码 worker。线程限制 2，`CUDA_VISIBLE_DEVICES=''`。TERM/INT 或截止时刻后，在当前单模型操作完成后停止，不删除 partial 草稿。原 CP、ZIP、恢复验收文件全部保留。每个模型首例报告 ZIP/raw 比率、encode CPU、完整模型往返 CPU/wall，以及明确标为进程累计的 peak RSS；各子计时不与父计时重复相加。

410M base 不重复保存：依赖已 durable 的 primary TAR，其完整 SHA `2bb090fbe7df172b31102068715143ab702087ff49f77ef7dfd5db3637c928b1`、当前 inode/dev/owner/permissions/size/mtime/ctime、历史已验收的小 manifest SHA、base 各文件/member SHA 与各 FP32 tensor 基底 SHA 全部写入依赖。此次只查历史完整验收＋当前 stat＋固定小 member，不宣称重新读取并验收整个 24.9GB TAR。1B/2.8B 则分别携带本模型的原始 pinned base、config/tokenizer/README 文件闭包，禁止跨模型引用。全部包携带独立离线 restore CLI/codec；optimizer 不包含，不承诺完整 Adam 恢复。

输出独立 scratch 模型目录与未压缩 TAR。预计 ceiling 为 410M 27GiB、1B 70GiB、2.8B 140GiB，实际超过即拒绝完整 TAR；这些估算并非压缩上界保证。现共享 reserve 路径仅写入预案，不由脚本打开或发布，报告 1+3GiB reserve 不动。完整 TAR 成功后仍需要 root 审核实际大小、SHA、验收及共享 reserve 的真实容量，再由既有 publisher 发布。

仅预检命令（默认不会编码）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src python scripts/export_C_final_weights.py
```

root 审核后才可启动的命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src python -u scripts/export_C_final_weights.py --execute --poll-seconds 30 --until 2026-11-07T02:00:00Z
```

进度为 scratch `delivery/extended_C_final_v1/progress.json` 与每模型 `draft_progress.json`；完整成功只以各模型 `completion.json` 为准。异常保留 `error.json` 和所有已有文件，需要 root 复核，不覆盖此前完整或 partial TAR。
