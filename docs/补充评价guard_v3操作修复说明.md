本次只修复补充评价解封前的独立门禁。C410 的完整模型状态证明中，`producer_source` 是 Python 源码的路径与 SHA，v2 将它当作 JSON 解析，导致第一次真实 check 失败。此次准备没有执行真实 check/analyze，没有读取评分语义，没有签发 TRUE authority 或创建真实 first-read journal。

最终实现为 `src/replay_study/supplemental_analysis_guard_v3.py` 和 `scripts/analyze_supplemental_phase_v3.py`。旧 v2 两个文件、协议、prepared plan、FALSE authority 模板均保持原字节；运行中的导出器和原 worker/collector 不变。`docs/supplemental_aux_analysis_operational_repair_v3_C410.json` 是非授权修复收据，精确绑定原 C410 plan、manifest、protocol 以及两代实现源码。初稿修复收据保留，但已在 `artifacts/provenance/supplemental_guard_v3_repair_v1/SUPERSEDED.json` 标明未用于真实操作且已被 plan-bound 版本替代。

v3 原样调用 v2 `load_plan`，因此原 preparation 源码 SHA、重新 prepare 的身份比较、checkpoint 白名单、显式 immutable cache snapshot 和 first-read 路径都保留。只有 `producer_source` 走 opaque SHA 绑定，不执行或 JSON 解析其内容；其他现有 audit/registry/snapshot 来源继续解析 JSON 并核验状态、唯一 ID、完整整数计数等结构。完整状态算法、指纹、双方 checkpoint SHA、原缓存五项 stat 和输出 attestation 校验仍保留。

所有 closure jobs（包括 imported initial/reuse）必须有一致 definition SHA；完成记录必须为 `type(returncode) is int` 且等于零，不能有 `stop_requested_at` 或 `stop_requested` 键，实际记录的 `argv` 必须等于冻结 wrapper 定义。历史 checkpoint 缓存的 completed authority 也拒绝布尔/浮点 rc0 和 stop 标记。评分文件仍只做 opaque SHA 校验，check 不解析其语义。

97 项合成测试通过，全部使用临时假数据。新 fixture 包含实际非 JSON `producer.py`、三种 JSON 来源、双方完整状态证明和 alias：v2 精确复现 JSONDecodeError，v3 通过；篡改 producer、JSON 结构、状态、argv、cache authority 或 repair scope 均被拒绝。新 analyze 主体去掉显式 repair 参数及 COMPLETE 收据新增字段后，AST 与 v2 完全相同；`write_auxiliary` AST 也相同，5000 次 bootstrap、20261106 seed、统计调用和循环顺序均未改。证据在 `artifacts/provenance/supplemental_guard_v3_repair_v2/`。

ROOT 可在独立 attempt 目录保留完整 spawn→exit wall/user/sys、stdout/stderr 后运行以下 check；这段命令没有由本次修复准备执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python scripts/analyze_supplemental_phase_v3.py check \
  --plan artifacts/analysis_prepared_v2/extended_pythia_410m_100m_nll.json \
  --plan-sha256 61f44eba006214462680f656fcb6cc3a90aae550ae506f84f2ec27446f2292fb \
  --operational-repair docs/supplemental_aux_analysis_operational_repair_v3_C410.json \
  --operational-repair-sha256 335a332d7337b3b11cba306722173e56ed7886f40cb9fd1dab3b457f6cae7e2d \
  --state logs/supplemental_job_state.json \
  --output artifacts/provenance/C410_nll_analysis_actual_v3/check_readiness.json
```

ROOT 需先核真实 check 的 `ready=true`、完整 closure 和 source/alias 输出完整性、checkpoint cache hit/fallback 明细。将来新 TRUE authority 必须保留原 FALSE 模板的全部绑定，并增加 `operational_repair` 路径与 SHA；旧模板不覆盖。修复收据本身的 `score_access_authorized=false`，不能授权评分分析。随后仅 ROOT 可运行新 CLI 的 analyze，并显式传入相同 repair 与新 authority。journal 路径仍取旧 plan 的 `first_read_receipt_path`，不能指定新目录来重置首次访问。

修复期间实测的 pytest whole-child timer 单独记录；它不代表整个编码、审阅或文档工作的时间。旧真实失败 check 的约 1.508 秒是独立失败尝试，不能被这次测试覆盖或改称成功分析成本。
