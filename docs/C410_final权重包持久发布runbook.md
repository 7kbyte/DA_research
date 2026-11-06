先定位以下路径；本轮只准备，没有发布或读取 TAR/checkpoint 大载荷。

| 项 | 路径 |
|---|---|
| 当前导出目录 | `/tmp/grad-research-20261106/delivery/extended_C_final_v1/pythia-410m/` |
| 最终 receipt / manifest | 上目录的 `completion.json` / `extended_C_final_manifest.json` |
| 最终源 TAR | 上目录的 `extended_C_410m_final_weights_lossless_v1.tar` |
| 27 GiB reserve | `/data/202610/7kb/research_delivery/.extended_C_410m_final_weights_reserve_20261106` |
| 预期持久目标 | `/data/202610/7kb/research_delivery/extended_C_410m_final_weights_lossless_v1.tar` |
| 旧容量 receipt | `artifacts/provenance/supplemental_weight_reservations_v1/C_410m_growth_receipt.json` |
| 整模型 READY | `artifacts/evaluation_ready_v5/extended_pythia_410m_100m_nll/READY.json` 及同目录 `manifest.json` |
| 全阶段 audit/proof | `results/extended_phase_equivalence/extended_v2_pythia_410m_m64_formal/{completion,audit,final_model_equivalence_proofs}.json` |
| 原 publisher | `scripts/publish_reserved_archive.py` 的 `publish()` / CLI |

reserve 已实分配 28,991,029,248 字节，dev106、inode484145481、uid/gid11471、mode0600，与旧 complete growth receipt 及 ROOT 全容量验收一致。只读 `/proc/locks` 未见该 reserve 被持锁，旧 grower 已结束。该观察不是锁 lease；执行前须再核同一身份。原 publisher 会自行以 nonblocking exclusive lock 获取 reserve，检查打开 FD 与路径一致、整个源 SHA、预分配长度及同文件系统。

全局 `/tmp/grad-research-20261106/delivery/.supplemental_weight_export.lock` 仍由 live daemon PID3168336 持有。不要抢它或要求 daemon 整体退出；C410 completion 形成后，daemon 可继续其他模型。发布不改 scratch 源包，原 publisher 独立获得 source shared lock 和 reserve exclusive lock，源路径/fd stat 全程稳定才提交。旧 grower未退出、reserve有锁、身份不同或目标已存在都应停止，不能覆写或临时更换 inode。

实际包完成后，ROOT 可先运行新的一次性 metadata reviewer（此处命令尚未执行真实 phase）：

```bash
python artifacts/provenance/C410_final_publication_runbook_v1/review_metadata.py \
  --output results/C410_final_publication_metadata_acceptance_v1.json
```

它只读小 JSON/小源码和 stat，不 import live exporter、不开 tensor、不读 TAR payload、不调用 publisher。会拒绝缺失 completion，要求真实全 24 阶段 READY/audit、同一 100M horizon 和 final checkpoint 身份、三 seed D/B4 final 全状态逐位证明、**24 logical / 21 unique / 3 D aliases**、每独立 weight 的完整 roundtrip 标志、每条 metadata roundtrip 和 rc0。若 D 不等价或物理数不同，这个预期21的 reviewer 会拒绝，须 ROOT 按真实来源另审，不强行 alias。也会核外部 durable base closure、原工具 SHA、源/ZIP/sidecar/primary TAR 的 stat、reserve owner/inode/实际容量、目标不存在，最后重核已读输入 stat。

`verified_tar_members`、TAR SHA、21 份 bytewise roundtrip 在这里是**原 exporter 的已完成证据**，metadata reviewer 不冒充再次逐 member/逐 tensor 验证。原 `build_verified_tar` 已逐 member SHA、全 TAR SHA、fsync，completion 的 `manifest_sha256` 必须匹配实际小 manifest。其原 builder、codec 和 publisher 源 SHA 也被 reviewer 固定。

ROOT 审核该新收据并明确决定发布后，才运行以下原 publisher 调用。实际 SHA/bytes 直接从已审核 metadata receipt 提取，不手抄、不占用新的 reservation：

```bash
python - <<'PY'
import json, subprocess, sys
from pathlib import Path
p = Path('results/C410_final_publication_metadata_acceptance_v1.json')
r = json.loads(p.read_text())
assert r['status'] == 'passed_metadata_only_for_ROOT_review_not_publication_authority'
subprocess.run([
    sys.executable, 'scripts/publish_reserved_archive.py',
    '--source', r['source'], '--expected-sha256', r['expected_sha256'],
    '--expected-bytes', str(r['expected_bytes']), '--reserved', r['reserved'],
    '--destination', r['destination'],
    '--receipt', 'docs/C410_final_weights_durable_publication.json',
    '--progress', 'artifacts/provenance/C410_final_weights_publication_progress.json',
], check=True)
PY
```

metadata 收据本身不构成发布 authority；ROOT 应沿旧 A 发布方式保存具体 acceptance/launch 决定，并在执行前再核 reserve 原身份。CLI 没有接收“计划 inode”参数，工具内部绑定的是它启动时的 reserve；因此不能省略外部身份复核。原工具会完整读源 SHA、复制到旧 inode、fsync、读回有效 prefix SHA、仅此后截短、原子 no-replace 提交与 parent fsync，保留源；不需要另做一遍 GB 级 `sha256sum` 才能启动。若失败，保留 progress/receipt 和 inode，不盲目重启或使用 overwrite。计时可用 Bash 内置 `time` 包住整个调用；publisher 自己的 wall 是父调用子项，不能双加。

成功后，ROOT 核发布 receipt 的 status=passed，source/destination SHA 与 completion 的 TAR SHA 一致，bytes 一致，destination 仍 dev106/inode484145481、uid/gid11471/mode0600，且 `original_inode_preserved`、`permissions_and_ownership_preserved`、`prefix_sha_verified_before_truncate`、file/parent fsync、`atomic_destination_commit_noreplace` 为真。允许已测 hard-link no-replace fallback；不能强求只有 renameat2。reserve 名应消失、持久目标存在、scratch 源保留。将 completion、manifest、READY/audit/final-proof、metadata review、publication receipt 的小文件 SHA 和副本存入 provenance/assets；不改原 exporter 输入或导出逻辑。

C410 包的 base 是 **durable external primary base closure**，不是 native base 已包含在本包。依赖为已持久发布 `primary_final_weights_lossless_v1.tar`（SHA `2bb090fbe7df172b31102068715143ab702087ff49f77ef7dfd5db3637c928b1`）中的 `base/pythia-410m-final/`；其 model base SHA 为 `3e09001631bb18ec8a6f4522f8e6ab26a53099e69228f9822adc91bfbe27511b`。metadata reviewer 使用历史完整验收加当前 primary TAR 身份/stat，不再次全读旧 TAR。最终恢复需要这两份 durable TAR，离开 scratch 后仍不需原 checkpoint、数据、网络或临时模型路径；**C410 TAR 单独不构成完整独立闭包**。

恢复示例仅供未来用户/ROOT 按需执行，本轮未执行：从 primary TAR 提取 `base/pythia-410m-final/`，从新包提取 tools、manifest、所选 `weights/RUN.xor.zip` 和相应 metadata，再运行

```bash
PYTHONPATH=tools/src python tools/scripts/archive_model_weights.py --threads 2 decode \
  --archive weights/RUN.xor.zip \
  --base base/pythia-410m-final/model.safetensors --output restored.pt
```

按 manifest 查 canonical weight member。D 使用对应 B4 weight 加自己的独立 source-metadata sidecar，保留各自 run 身份。decoder 核 base/tensor/full-model hashes；将来独立恢复 QA 必须另留实际 timer/指纹收据，不能把本次 metadata 准备写成已恢复。该交付不含 optimizer，不承诺完整 Adam resume。completion 的 assembly-only timer 与先前 draft roundtrip 独立，后续 publication timer 再独立登记，daemon 累计等待不重加。
