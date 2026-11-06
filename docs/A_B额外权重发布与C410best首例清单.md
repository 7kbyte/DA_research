# A/B 额外权重发布与 C410 best 首例清单

本清单是 CPU 准备，不是发布或 C 样例许可。运行中的 v4 exporter、原 C/A exporter、codec、helper、SLOTS 和锁均未改。来源与当前观察见 `artifacts/provenance/additional_endpoint_publication_runbook_v1/preparation_receipt.json`；A/B 仍在编码，真实 completion/whole-batch timer 尚未产生，不能补造验收。

独立 metadata reviewer 已按 ROOT 审阅补强到 revision v2：额外核 config/summary 与 scope 原 input SHA，一律要求严格整数 rc0，两个 stop 键出现即拒绝。当前源码/13项小型测试及旧版证据见同目录 `reviewer_revision_v2.json`；旧准备收据保留原字节。该更新不改任何 live exporter 或其 imported helper。

## A/B 完成后，先 metadata 验收，再由 ROOT 单独发布

scratch 根目录固定为 `/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/`。

| 项目 | A_remaining_final | B_all_final |
|---|---|---|
| 完整范围 | 93 logical / 93 unique final | 60 logical / 60 unique final |
| 源 TAR（各自子目录内） | `A_remaining_final_weights_lossless_v1.tar` | `B_all_final_weights_lossless_v1.tar` |
| shared reserve 后缀（同 delivery 目录） | `.extra_A_remaining_final_weights_reserve_20261106` | `.extra_B_final_weights_reserve_20261106` |
| inode（dev 均106） | 622542307 | 411787337 |
| 实配 bytes / GiB | 123,480,309,760 / 115 | 80,530,636,800 / 75 |
| whole-batch timer | `artifacts/provenance/additional_endpoint_actual_v4/A_remaining_final_batch/whole_child_timing.json` | 同目录下 `B_all_final_batch/whole_child_timing.json` |

两 reserve 当前 owner/group 均11471、mode0600、nlink1，实际分配字节等于表内长度，与 ROOT `ROOT_full_capacity_acceptance.json`（SHA `a1532181…`）相同；`/proc/locks` 当时未见 reserve 锁。该观察不是 lease，ROOT 启动 publisher 前须重新确认身份和容量，不能换 inode。目标文件应位于 `/data/202610/7kb/research_delivery/`，文件名与源 TAR 相同，必须不存在。报告 reserve 及其他文件不参与。

逐包等待 **completion 和 whole-batch timer 都真实存在** 后，可运行下面的新薄 metadata 入口。缺 completion 直接拒绝且不产生“通过”收据；尚未执行真实完整 A/B 验收。

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH=src python artifacts/provenance/additional_endpoint_publication_runbook_v1/review_metadata.py \
  --package A_remaining_final --output results/A_remaining_final_publication_metadata_acceptance_v1.json
```

B 改为 `--package B_all_final --output results/B_all_final_publication_metadata_acceptance_v1.json`。入口复用冻结 v4 的纯 metadata validators，不调用 `load_scope/prepare/execute`。只读最大32MiB的小 JSON/源码/sidecar及文件 stat，不读或 SHA CP、ZIP、TAR、native model 载荷。它要求：

- 源码 `c93e0c1d…`、scope v4 `7adb8f7f…`、实际 batch authority `d73a773e…`；完整93/60范围、全部源身份/配置/summary/训练 job definition/实际 argv/rc0/no-stop，允许其他评价合法 append。
- 每条 original record 与 manifest 一致；源 CP SHA 与冻结身份一致；292 tensors、405,334,016 elements 的完整 fingerprint 在 encode/decode/逐字节记录一致；所有原 uint8 与非权重 metadata 往返完成；源 path/fd 五项 stat 前后一致，当前 CP/ZIP stat 不变。
- 每包自己的六个 native base/config/tokenizer/README 成员、tools、provenance、所有 metadata 和 ZIP 构成独立恢复闭包；不依赖旧 primary 或未来 C final 包。base SHA为 `3e09001631bb18ec8a6f4522f8e6ab26a53099e69228f9822adc91bfbe27511b`；不含 optimizer。
- 原 builder 的全部成员/全 TAR SHA 与 fsync 完成、manifest SHA一致、真实 TAR stat/实际长度不超 reserve；本轮只核已有证据，不称再次完整 TAR/tensor 验证。whole-batch child 已rc0，reserve身份/owner/capacity仍一致、无观察到的锁，目标不存在。

ROOT 审核通过、保存具体 publication authority 后，直接复用 **C410 已用的原 publisher**。以下代码只展示待授权命令，不运行；B 只替换 `name`。

```bash
python - <<'PY'
import json, subprocess, sys
from pathlib import Path
name='A_remaining_final'
r=json.loads(Path(f'results/{name}_publication_metadata_acceptance_v1.json').read_text())
assert r['status']=='passed_metadata_only_for_ROOT_review_not_publication_authority'
subprocess.run([sys.executable,'scripts/publish_reserved_archive.py',
 '--source',r['source'],'--expected-sha256',r['expected_sha256'],
 '--expected-bytes',str(r['expected_bytes']),'--reserved',r['reserved'],
 '--destination',r['destination'],'--receipt',f'docs/{name}_weights_durable_publication.json',
 '--progress',f'artifacts/provenance/{name}_weights_publication_progress.json'],check=True)
PY
```

发布前无需再执行一次 GB 级 `sha256sum`：publisher 会新核完整源 SHA、用旧 reserve inode 流写、fsync、读回完整有效 prefix SHA，成功后才截尾、原子 no-replace commit、父目录 fsync；源保留。必须重新核计划 reserve 身份，因为 CLI 自己绑定的是启动时 inode，并不接收预期 inode 参数。允许已验收 hard-link fallback。发布后核 `passed`、源/dest SHA/bytes、原 inode/owner/mode 保留、prefix-before-truncate、file/parent fsync、no-replace；stat-only复核不可冒充再次全包 SHA。

whole-batch user/sys/wall 是该批 startup、resume 首例、后续编码和 assembly 的 inclusive 父项，不与 endpoint/assembly 子项相加。先前各一次样例是另外的已结束 child；publication 与 metadata review 再独立计时。可用 Bash 内置 `time` 或 ROOT 既有 Python wrapper，平台没有 `/usr/bin/time`。

## C410 best：资料实际齐，但仍等 A slot1 与 ROOT 许可

实际来源如下；小文件 fresh SHA 和逐项 source bindings 已写入 preparation receipt，不打开 checkpoint 或评分文件。

- `artifacts/evaluation_ready_v5/extended_pythia_410m_100m_nll/READY.json`：SHA `d776f2aa…`，绑定同目录 manifest SHA `48f8b2235c303211e4f79a92c16c7fb1c8d3757ce5d66a97a555fba7dfcf8f32`；24 final +24 best 的身份记录，实际只取24 best。
- `results/extended_phase_equivalence/extended_v2_pythia_410m_m64_formal/completion.json` 与 `audit.json`：24/24、passed、无 source violations、真实训练 rc0/no-stop/argv/config/冻结 source 通过。
- 同目录 `best_model_equivalence_proofs.json`：SHA `423be07d69fe5157541bd82896077cdf6de839aebde54e5deb1913843f0d4303`。这是三 seed 的 **3对、6个 source-bound best 端点**，不是6个独立重复；三对均 bitwise true，FP/CP SHA/配置/summary/实际曝光绑定当前 READY，支持未来24 logical /21 unique /3 D aliases。未用 phase passed 代替逐对 true。selection 数值仅 opaque hash，不解读。

**固定 SLOTS 为 slot1:A→C410→C1B、slot2:B→C2.8。B 完成不能解锁 C410；必须等 A 释放 slot1。** 不改锁、不从 slot2 绕过。ROOT 以后明确允许首例后，原 argv 为：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH=src python scripts/export_additional_endpoints.py execute \
  --scope /tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/scope.json \
  --scope-sha256 7adb8f7f0461fd929995dd370ff2a59b2abc5c0d7750a39e745d21d5546d9de3 \
  --slot slot1 --package C410_best --max-new-weights 1
```

按冻结排序，首例将是 `extended_v2_pythia_410m_m64_formal_B0_s11_w0` 的 best，源 CP SHA `dd25db838ede6d137cd630818e5ecb3ad8f9100ef47eba86740a48f04a260bf4`；它的 fresh model FP与资源 receipt仍待实际样例，不拿 B4 proof 冒充 B0 FP。既有 A/B 批 authority 不授权此调用；max1 不需要也不应借用 A/B resource-review。执行会重新核输入并新读该 CP，超出本页只读 metadata 准备。

首例完成后，ROOT 独立核 whole-child/endpoint wall、CPU、累计 RSS、共享 cgroup events、源/base/fingerprint/全 uint8/metadata 往返；再单独决定 C410 批量。届时剩20 unique 编码和3 D 独立源身份/metadata核验，后者不能当零工作。whole-phase投影须包含保守 per-item、别名核验/assembly、publication 与 margin：预计 publication start严格<02:00，再加publication与margin≤02:56:32；实际02:00不新开任何操作。未形成实际样例前，本页不提供 TRUE resource许可或“已可完成”结论。C410 best reserve为30GiB（32,212,254,720B）、dev106/inode467366543、owner/group11471、0600，当前未消费；是否发布仍另经完整包与 ROOT 验收。
