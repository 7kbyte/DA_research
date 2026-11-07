# A behavior 原 v2 完整分析：ROOT 操作清单

此次只准备元数据与命令，未运行 check/analyze、创建 TRUE、复制 state 或读取 A 行为评分。原 plan/协议/guard/collector 一律不改。A 使用原 v2；不要借用 C 的 v3 repair。

22 项实际闭包已全部 completed，严格整数 rc0、两个 stop 键不存在、无 missing、实际 argv 与冻结 definition SHA 相等。闭包由 15 个新增规范 forward、3 个 imported B1 forward、1 个 imported matched initial 和 3 个 reuse 构成：19 canonical+3 reuse=22；18 个 selected final 逻辑端点。不能只看本 A phase 的18项。六方法 λ 固定为 B1=.05/B2=1/B3=.1/B4=5/M=5/C=1，来自原 validation selection SHA9548085fc18278f69fad584b67485bceb5193a18b10c78f516f6b09b4fa4e477；不得用行为重选。

ROOT 已实际完成 immutable state capture，不再使用 pending/live state，也不重新捕获：

- state：`artifacts/provenance/A_behavior_analysis_actual_v1/state_capture/actual_successor_state_immutable.json`
- SHA：`27b41e7ee3d8f7e49180e3bf228684816a15d6f7558df7ea65c40624c1eadb88`
- 相邻 `ROOT_state_capture_receipt.json` 是实际捕获证明；本次仅再次读取元数据核22闭包，没有复制 state。
- 此前 metadata_closure_observation.json 的 live-state SHA 是历史观察，不能代替上述实际 immutable SHA。先前 `future_commands_PREPARED_ONLY.json` 保留 pending-state 历史；执行请用新 `future_commands_ACTUAL_STATE_PREPARED_ONLY.json`。

以下均为尚未由本代理执行的 ROOT 命令。在 `/tmp/grad-research-20261106/worktree` 运行，CVD 空、OMP/MKL/OPENBLAS 两线程。ROOT 为 check 与 analyze 各保留独立完整 Popen/resource spawn→exit 计时、stdout/stderr，已有目录/receipt 不覆盖。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/usr/bin/python scripts/analyze_supplemental_phase_v2.py check \
  --plan /tmp/grad-research-20261106/worktree/artifacts/analysis_prepared_v2/supplement_A_behavior.json \
  --plan-sha256 7ab89cc4fc71287d3df8b04098c51db92a4720ae10eff36ed421f5b44d82ed8a \
  --state /tmp/grad-research-20261106/worktree/artifacts/provenance/A_behavior_analysis_actual_v1/state_capture/actual_successor_state_immutable.json \
  --output /tmp/grad-research-20261106/worktree/artifacts/provenance/A_behavior_analysis_actual_v1/check_readiness.json
```

check 必须实际 strict0、`ready is true`、`pending_job_ids=[]`、closure22，`scheduler_state` 精确等于上述路径，`scheduler_state_sha256` 等于27b41e…；plan/counts/cache/source/输出 attestation 都须原 guard 通过。它会执行原 source、CP cache 与输出字节验证；本次 metadata 观察没有替代这些工作。若失败保留证据，不能改原 plan 或换较宽门禁绕过。

ROOT 审阅实际 check 后，才从原 `artifacts/analysis_prepared_v2/supplement_A_behavior_root_authority_TEMPLATE.json`（SHAe88c6cd50d38d27cf41a55e8c139a85f3afba356e943fdbd3597a64de367b9ef）写一个全新 `artifacts/provenance/A_behavior_analysis_actual_v1/ROOT_A_behavior_authority.json`：

- 仅在实际批准后，把 `allow_score_analysis` 设为精确布尔 true；原 FALSE 模板保持原字节。
- 原 `authorized_by=root`、plan/protocol/READY/manifest 路径与 SHA、contract_id、expected_counts、checkpoint_verification、`allow_partial=false`、训练冻结与未用行为选参字段全部保留。
- 附加 exact immutable state path/SHA 和实际 check path/SHA 作 ROOT 操作证明。原 v2 没有额外 state CLI SHA 参数，故这是外部 ROOT 执行要求；不得声称新核心门禁。
- state、check 和 TRUE 各自 fresh SHA 后再分析，state 从首次 check 到 analyze 不变。不要用 NLL authority，也不要复用主实验行为 authority。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/usr/bin/python scripts/analyze_supplemental_phase_v2.py analyze \
  --plan /tmp/grad-research-20261106/worktree/artifacts/analysis_prepared_v2/supplement_A_behavior.json \
  --plan-sha256 7ab89cc4fc71287d3df8b04098c51db92a4720ae10eff36ed421f5b44d82ed8a \
  --authority /tmp/grad-research-20261106/worktree/artifacts/provenance/A_behavior_analysis_actual_v1/ROOT_A_behavior_authority.json \
  --authority-sha256 ROOT_ACTUAL_TRUE_AUTHORITY_SHA256 \
  --state /tmp/grad-research-20261106/worktree/artifacts/provenance/A_behavior_analysis_actual_v1/state_capture/actual_successor_state_immutable.json \
  --output /tmp/grad-research-20261106/worktree/results/supplemental_analysis_v2/supplement_A_behavior
```

上面 TRUE SHA 只能由 ROOT 的实际新文件填入，本清单不创建它。analyze 会重新执行冻结 inspect；两次执行分别计费，内部子计时不重加。保留完整 final+validation_selected_policy 原输出；各方法不同 λ 是验证选 λ 后的策略对比，不伪装成固定 λ 的 objective 对比。以后独立 reviewer 会处理原 selected-policy 聚合 CSV 的 score 标签边界，不改原 producer。

首次语义读取 journal 原 plan 已固定为：

`artifacts/provenance/supplemental_score_first_reads_v1/supplement_A_behavior_31f0877c6153a4a4539ca60dc4e2e666bb32d74416de9fd5711cdd65a10848fa.json`

本次预查时该文件不存在。不得传替代 first-read 路径、换输出目录重置 journal 或接受 partial。ROOT 实际成功 COMPLETE 与完整父 timer strict0 后，才进入获授权的 A 独立数值 reviewer；本次没有运行它。当前元数据短父计时和新增 actual-state binding region 各保留，人工阅读/编写的完整墙钟未知。
