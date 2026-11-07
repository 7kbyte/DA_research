# 补充分析 checkpoint 核验缓存：操作修订 v2

本修订只减少补充分析门禁的重复 checkpoint 字节读取。选择、统计公式、5000次bootstrap及seed、打印图尺寸、GPU forward范围、全phase完整性与root显式授权要求均不变；不读取任何真实补充或行为分数，不启动GPU或追加队列。主primary行为分析guard及8个冻结评分文件保持原字节。

v1原脚本/模块/协议/命令/验收均留在原路径，并有完整字节副本与SHA清单：`artifacts/provenance/supplemental_aux_analysis_v1/source_bytes_manifest.json`，SHA `49eab2b0b24e65904d9131147338149f0e90cc26d81768688b14220cb5804583`。v1未读取真实分数。v2新增独立 `_v2.py` CLI和guard，没有monkeypatch或替换冻结函数。

冻结v2协议：`docs/supplemental_aux_analysis_protocol_v2.json`，SHA `eed050602c368e5e4be87172ac4d208b890d9a7f7e9811ce1684ae62acc8f742`。命令模板：`configs/supplemental_aux_analysis_commands_v2.json`，SHA `fb46ee08de3c578b0ba23d14f1d5191d8653bda66169f976f1b0aa7437b7765c`。

## 显式模式与边界

`prepare` 不带cache参数时，计划明确记为 `checkpoint_verification.mode=fresh`，checkpoint全部fresh hash。可选cache必须同时提供准确snapshot路径和SHA；不自动寻找“最新snapshot”，也不会从环境变量隐式打开cache。每个phase必须先有真实READY，再选择一个immutable v5 snapshot。v5 `cache_progress.json` 的 `cache_snapshot` 字段只是候选proof，尚须核对其文件SHA并显式传入。

只有来自SHA绑定manifest的实际 checkpoint 路径，以及其中完整model_state等价证明的双方checkpoint路径，才进入独立allowlist；不能凭`.pt`扩展名让evaluation tensor命中。source code、manifest/protocol/config/summary等JSON、score output、attestation及prediction JSONL均在本次guard调用fresh验证；不能因这些文件恰好出现在snapshot中而使用历史摘要。

命中要求全部满足：snapshot SHA及受信v5 reader source SHA一致；receipt自身SHA与schema正确；历史完整读取达到EOF，read_bytes等于文件大小，path/fd的before/after五项stat完全一致；receipt来源是成功完成的training checkpoint；当前canonical路径、dev/ino/size/mtime_ns/ctime_ns五项stat与receipt相同；历史SHA等于当前phase冻结expected SHA。结束时再次核对文件stat和原始symlink解析目标。snapshot缺项或没有合格checkpoint来源receipt时fresh回退；stat变化会通过受信reader完整重读再校验，已完成checkpoint内容改变则拒绝。坏snapshot、坏receipt、错误expected SHA均拒绝。

allowlist、明确fresh/cache模式、snapshot proof和reader proof保存在阶段plan；root授权除plan/protocol/READY/manifest/完整counts外，还必须精确绑定 `checkpoint_verification`。换snapshot或模式须重新prepare并由root独立审查，不能沿用旧授权。全phase含prior baseline、canonical forward、reuse和logical aliases仍须completed/rc0、所有源与输出SHA完整。C的final/best共同过门禁。

首次语义解封回执继续使用v1同一个 **contract ID + manifest SHA固定路径**，因此不能通过切换操作版本重置首次时间。先保存完整readiness与不可覆盖first-read receipt，才调用既有collector。cache不会扩大可读阶段或允许partial结论。

## 命令

使用v2的9份结构化模板，默认fresh。真实READY产生后，如明确选择cache，在模板的prepare命令上添加：

```bash
python scripts/analyze_supplemental_phase_v2.py prepare \
  --ready artifacts/evaluation_ready_v5/supplement_A_nll/READY.json \
  --ready-sha256 '<READY_SHA256>' \
  --protocol docs/supplemental_aux_analysis_protocol_v2.json \
  --protocol-sha256 eed050602c368e5e4be87172ac4d208b890d9a7f7e9811ce1684ae62acc8f742 \
  --contract-id supplement_A_nll \
  --digest-cache-snapshot '<EXACT_IMMUTABLE_V5_SNAPSHOT_PATH>' \
  --digest-cache-snapshot-sha256 '<SNAPSHOT_SHA256>' \
  --output artifacts/analysis_prepared_v2/supplement_A_nll.json
```

去掉两个`--digest-cache-*`参数即fresh模式。prepare生成的root authority TEMPLATE仍为false，不能直接当解封授权。`check`/`analyze`只消费plan中冻结的模式，没有临时替换snapshot参数。示例：

```bash
python scripts/analyze_supplemental_phase_v2.py check \
  --plan artifacts/analysis_prepared_v2/supplement_A_nll.json \
  --plan-sha256 '<PLAN_SHA256>' \
  --state '<STATE_WITH_FULL_ORIGINAL_AND_NEW_CLOSURE>' \
  --output results/supplemental_analysis_v2/supplement_A_nll_readiness.json

# 只在root另行准确授权后运行。
python scripts/analyze_supplemental_phase_v2.py analyze \
  --plan artifacts/analysis_prepared_v2/supplement_A_nll.json \
  --plan-sha256 '<PLAN_SHA256>' \
  --authority '<ROOT_AUTHORITY_JSON>' \
  --authority-sha256 '<ROOT_AUTHORITY_SHA256>' \
  --state '<STATE_WITH_FULL_ORIGINAL_AND_NEW_CLOSURE>' \
  --output results/supplemental_analysis_v2/supplement_A_nll
```

## 成本字段与只读collector审查

readiness和COMPLETE记录 `verification_work`：`checkpoint_cache_hits`、`checkpoint_cached_bytes`表示历史完整receipt覆盖的当前checkpoint大小，**本次读取0个payload字节**；`checkpoint_fresh_reads/bytes`才是本次guard的完整checkpoint读取。`checkpoint_events`逐路径绑定expected SHA、5stat、历史receipt proof及本次current_bytes_read，缺项回退和stat失效均可追溯。

`other_fresh_reads/bytes`只计此Verifier显式hash的独立非checkpoint路径，不冒充cache library、嵌套prepare或collector所有小文件读取的全研究I/O总量。`within_invocation_reuses`是同一次Verifier已fresh验证且stat未变时的重复引用。cache reader自身counter是子诊断，与guard的checkpoint fresh字段描述同一次读取时不可重复相加。whole verification wall不是GPU时长，命中stat检查、源/结果fresh核验和CPU统计/出图仍有真实成本，未测部分不写0。

只读源代码审查确认：

- `nll_v2_analysis.prepare_plan` hash实现、manifest/spec/registry及model-state proof JSON；checkpoint路径/SHA只作身份元数据，不打开checkpoint payload。
- `collect_scores`前后复核其metadata/output来源，解析完整attested NLL结果，bootstrap仅使用结果内的chunk/document统计；不hash或反序列化checkpoint。合成集成测试禁止其`sha()`接触任一checkpoint路径，完整collector仍通过。
- `scripts/analyze_behavioral_plan.py:collect` hash manifest/common inputs、attestation、summary与prediction；调用冻结wrapper的`verify_outputs`与`validate_result`，不会调用重新hash训练输入的`verified_job`。冻结builder的common inputs不包含checkpoint；v2 guard还显式拒绝把allowlist checkpoint置入common sources，防止隐藏重复读取。
- 实际GPU评分worker的fresh完整输入SHA政策不变。本修订只作用于未来CPU补充分析guard，不减少已冻结GPU评分的来源校验。

相应来源SHA与审查行号记录在 `docs/supplemental_aux_analysis_collector_io_review_v2.json`。A的selected18仍严格复用全126已有结果，不新增forward，不重选λ；各自验证选λ后的策略比较及条件bootstrap解释边界沿用v1。
