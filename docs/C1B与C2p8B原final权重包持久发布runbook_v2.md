本清单仅用于原 Cfinal daemon 的 1B/2.8B 整阶段 **final** 包，不用于新增 Cbest 包。当前两包 completion、manifest 和 TAR 尚未形成，以下命令均为未来 ROOT 操作步骤；此次没有实际整包验收、发布或新增 authority。

| 项目 | 1B 原 Cfinal | 2.8B 原 Cfinal |
|---|---|---|
| scratch 源目录 | `/tmp/grad-research-20261106/delivery/extended_C_final_v1/pythia-1b/` | 同根下 `pythia-2.8b/` |
| TAR | `extended_C_1b_final_weights_lossless_v1.tar` | `extended_C_2p8b_final_weights_lossless_v1.tar` |
| 原 reserve | `/data/202610/7kb/research_delivery/.extended_C_1b_final_weights_reserve_20261106` | 同目录 `.extended_C_2p8b_final_weights_reserve_20261106` |
| 原实配 ceiling | 70 GiB = 75,161,927,680 B | 140 GiB = 150,323,855,360 B |
| dev / inode | 106 / 663438738 | 106 / 559646871 |
| owner / mode | uid/gid 11471 / 0600 | uid/gid 11471 / 0600 |
| 冻结实际 final exposure | 每项 100M target | 每项 25M target |
| 完整物理训练条件 | 24 logical；D/B4 三对实际证明相等后 21 unique ZIP + 3 aliases | 15 logical / 15 unique，无 D 条件 |
| 实际 whole-phase READY 名 | `extended_pythia_1b_100m_nll` | `extended_pythia_2p8b_25m_nll` |

两条 reserve 当前均仍原 inode、完整 allocated bytes、单 link，未观察到 reserve 锁；完整旧 grower receipt 在 `artifacts/provenance/supplemental_weight_reservations_v1/C_{1b,2p8b}_growth_receipt.json`。它们与新增 best 包 reservations 是不同对象。观察不是 lease，执行前须再核原身份和实配容量、旧 grower 已退出、目标不存在。不能因目标较小而换 inode、重新分配 reserve 或覆盖已有目标。

复用原 `scripts/publish_reserved_archive.py`。原 C410 metadata reviewer 固定了 C410 phase、24→21、27GiB 与 external-primary base，不能直接用于这两包。独立的 `artifacts/provenance/C_native_final_publication_runbook_v2/review_metadata.py` 只读小 metadata/源码和 payload stat，不 import 或改 live exporter、codec/publisher，不读 tensor/TAR/ZIP 内容，也不解析评分。它从 logical sidecars、unique ZIP、6 个 native base 文件、固定工具/provenance 和实际 gate sources 推导全部 TAR member 数，不套用 145；最终仍依赖原 exporter 的 all-member SHA、全 TAR SHA 和 roundtrip 收据，publisher 将 fresh 完整核验源和写入 prefix。

实际验收必须同时具备：

- 对应模型 whole-phase 的真实 READY/manifest、固定 final exposure、全部原 registry 条件，以及原 `fullphase_gate.status=complete_source_gate_passed`。不能凭部分 drafts 发布。
- 1B 的 per-model completion/audit、三 seed final 全模型指纹/双方 checkpoint SHA。21 unique 仅在三对 actual D/B4 proof 都严格为 true 时成立；任一不同则当前 21-unique reviewer 拒绝，由 ROOT 按真实独立编码 scope 另审，不强行 alias。2.8B 使用包内 `fullphase_source_audit.json` 的完整 15-run audit，不要求不存在的 D 证明。
- 每个 unique 权重实际 encode→decode→完整 uint8 比对、每个 logical 条件完整 nonweight metadata roundtrip、固定 definition/实际 argv、整数 rc0、无 stop 标记、config SHA 同时对冻结 snapshot、READY input SHA 和当前小文件 fresh SHA，summary SHA 对冻结 READY 与当前小文件 fresh SHA；checkpoint 规范路径同时对原 registry/冻结 final identity，实际 size/mtime 对冻结 identity，另核 row 自身 source stat。summary 只散列字节不解析评分语义，checkpoint 不读 payload。独立 unique draft 的 row FP、CP/ZIP/base SHA、ZIP路径/大小、decode 全 tensor hash 状态/数量与 encode/decode/bytewise 报告必须一致，拒绝拼接不同报告。optimizer 不包含，不能宣称完整 Adam resume。
- `completion.json` 与 `extended_C_final_manifest.json` SHA 绑定，完整 TAR member 数与实际 inventory 一致、全 TAR SHA、file/parent fsync 为 true，包体不超过原 ceiling，源包和中间产物保留。
- native dependency 精确覆盖该模型冻结 snapshot 的 `model.safetensors/config/tokenizer/special_tokens_map/README` 六个文件、SHA/bytes/revision 与 member 路径。1B 原 base SHA `7e6dff44849e036a73296368e663bd9420009977f03a8d4d747b234264f37107`；2.8B 为 `ab496f1c3fd79e3c749a9d5414136a2c8e4224f94eecb261970315cdb0f813fe`。本 reviewer 不 fresh 重读这些 GB base，原 exporter 完整 all-member SHA 已覆盖，publisher 再读完整 TAR。

ROOT 仅在上述真实文件齐全后运行以下 metadata reviewer，并为每次调用保存独立 spawn→exit wall/user/sys、stdout/stderr。两条命令此次均未执行：

```bash
CUDA_VISIBLE_DEVICES='' python artifacts/provenance/C_native_final_publication_runbook_v2/review_metadata.py \
  --model pythia-1b --output artifacts/provenance/C1B_final_publication_actual_v1/metadata_review.json
CUDA_VISIBLE_DEVICES='' python artifacts/provenance/C_native_final_publication_runbook_v2/review_metadata.py \
  --model pythia-2.8b --output artifacts/provenance/C2p8B_final_publication_actual_v1/metadata_review.json
```

metadata 收据只是 ROOT 审阅材料，不是发布授权。ROOT 另存 acceptance/authority 和具体 source/review/receipt SHA 后，可对单模型使用下面相同原 publisher 调用；将参数路径替换为该模型的实际 metadata review 和新 publication receipt。必须再核同一 reserve 身份，不能省略这一步，因为原 publisher CLI 不接收预设 inode 参数。

```bash
TIMEFORMAT='wall=%R user=%U sys=%S'
time CUDA_VISIBLE_DEVICES='' python - \
  artifacts/provenance/C1B_final_publication_actual_v1/metadata_review.json \
  docs/C1B_final_weights_durable_publication.json \
  docs/C1B_final_weights_durable_publication.progress.json <<'PY'
import json, os, sys
r = json.load(open(sys.argv[1])); receipt = sys.argv[2]; progress = sys.argv[3]
assert r['status'] == 'passed_metadata_only_for_ROOT_review_not_publication_authority'
argv = [sys.executable, 'scripts/publish_reserved_archive.py',
        '--source', r['source'], '--expected-sha256', r['expected_sha256'],
        '--expected-bytes', str(r['expected_bytes']), '--reserved', r['reserved'],
        '--destination', r['destination'], '--receipt', receipt, '--progress', progress]
os.execv(sys.executable, argv)
PY
```

上面 `time` 是 Bash 内置父计时，包住 metadata launcher 与 publisher exec；ROOT 应把实际 wall/user/sys 保存为 whole-child 口径，内部 publisher wall/readback/copy/fsync 都是嵌套子项。它不是此次已执行的命令或费用。原 publisher 自行取得 source shared lock 和 reserve nonblocking exclusive lock，完整 source SHA、复制/fsync、完整有效 prefix SHA、校验后截短，再 no-replace commit 和 parent fsync。保留 scratch source；不需先另外执行 GB `sha256sum`。失败保留原 receipt/progress/inode，不盲目覆写重试。

发布后只读验收 receipt：status passed、源/目标 SHA 和 bytes 等于已审 completion，目标 inode 仍是对应原 reserve、owner/mode 保留，`original_inode_preserved`、`permissions_and_ownership_preserved`、prefix-before-truncate、file/parent fsync、`atomic_destination_commit_noreplace` 为真。允许原已审 hard-link fallback。reserve 名消失、正式目标存在且 scratch 源仍在。ROOT 保存 completion、manifest、READY/source audit/1B final proof、metadata review、publication 和完整父计时的小字节 SHA/副本；不得把 reserve 容量或 assembly 成功提前写成 publication 成功。

原 Cfinal daemon 同时服务三个模型，其累计 wall 含 prehash、poll/锁等待、draft 与 assembly，不能视为孤立模型纯工作或与 children 再加。每个 first-created draft 的独立 wall/CPU 与每模型 completion 的 **assembly-only** wall/CPU 是不重叠操作；encode/decode/verify 是 draft 内部子项，复用 draft 不重新编码或重复计时。未来 publisher whole-child 是独立操作，内部 publisher wall 是其子项。

C410 的原 final TAR 使用已 durable 的 `primary_final_weights_lossless_v1.tar` 中 external-primary base，必须两包合用才可离开 scratch 恢复。1B/2.8B 原 final TAR 的 native base/config/tokenizer/README 在各自包内；包含恢复工具和 metadata，恢复模型无需原 checkpoint、scratch base、训练数据或网络，仍需常规 Python/Torch/codec 依赖。此次 metadata 准备不冒充新增实际 restore/decode 验证；完整 roundtrip 由原 exporter 真实记录提供。

本 v2 保留原 v1/95e 初稿及其准备收据，只修复未实际使用的独立 reviewer 和未来命令；原 v1 publisher 示例缺必需 `--progress`，不应用于执行。ROOT 对两模型均应使用各自独立 progress 路径。此次未运行实际整包 reviewer 或 publisher。
