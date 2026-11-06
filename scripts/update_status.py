#!/usr/bin/env python3
"""Render a compact local status page from the saved experiment records."""
import collections
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
now = dt.datetime.now(dt.timezone.utc)
deadline = dt.datetime(2026, 11, 7, 3, 26, 32, tzinfo=dt.timezone.utc)
started = deadline - dt.timedelta(hours=24)
protocol_path = ROOT / 'docs/protocol_status.json'
protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
lines = ['# 研究运行状态', '', f'更新时间：{now.isoformat()}',
         '截止时间：2026-11-07 03:26:32 UTC（北京时间11:26）',
         f'已用 {(now-started).total_seconds()/3600:.2f} 小时；剩余 {max(0,(deadline-now).total_seconds()/3600):.2f} 小时。',
         '', '以下状态来自运行日志。配置完成不等于实验完成；正式结论以最终研究报告为准。',
         '', f"当前阶段：{protocol.get('stage', '进行中')}", '', protocol.get('note', ''),
         '', '| 调度阶段 | 完成 | 运行 | 待执行 | 失败/阻塞 |', '|---|---:|---:|---:|---:|']
for path in sorted((ROOT/'logs').glob('*job_state.json')):
    try:
        state = json.loads(path.read_text())
        c = collections.Counter(x['status'] for x in state['jobs'].values())
        label = path.stem + ('（已隔离）' if path.stem in protocol.get('provisional_schedulers', []) else '')
        lines.append(f"| {label} | {c['completed']} | {c['running']} | {c['pending']} | {c['failed']+c['blocked']} |")
    except (ValueError, KeyError):
        lines.append(f'| {path.stem} | 状态暂不可读 | | | |')
phase_index = ROOT / 'configs/supplemental_v1_registry_index.json'
master_state = ROOT / 'logs/supplemental_job_state.json'
if phase_index.exists() and master_state.exists():
    index = json.loads(phase_index.read_text())
    jobs = json.loads(master_state.read_text())['jobs']
    lines += ['', '## 当前单一调度器内的阶段', '',
              '以下为上述 supplemental 总行的分解，不另加为新的任务；按冻结 job ID 统计。', '',
              '| 阶段 | 完成 | 运行 | 待执行/尚未登记状态 | 其他状态 |',
              '|---|---:|---:|---:|---:|']
    for name, phase in index.get('phases', {}).items():
        counts = collections.Counter(jobs.get(job_id, {}).get('status', 'not_registered')
                                     for job_id in phase.get('job_ids', []))
        other = sum(v for k, v in counts.items()
                    if k not in {'completed', 'running', 'pending', 'not_registered'})
        lines.append(f"| {name} | {counts['completed']} | {counts['running']} | "
                     f"{counts['pending'] + counts['not_registered']} | {other} |")
first_read = ROOT / 'docs/primary_nll_first_test_read.json'
lines += ['', '## 独立评价读取状态', '']
if first_read.exists():
    receipt = json.loads(first_read.read_text())
    lines.append(f"主 NLL 首次读取时间：{receipt['first_real_test_semantic_read_utc']}；"
                 '该授权只覆盖主 NLL，其他阶段和行为分数须各自满足解封条件。')
else:
    lines.append('主 NLL 已排队，只有全部 67 项完成并通过完整性门禁后才读取；尚无首次读取记录。')
lines += ['', '## 当前数据与选参记录', '']
for rel in ['artifacts/data/VERSION', 'artifacts/evaluation_v3/VERSION']:
    path = ROOT/rel
    if path.exists():
        lines.append(f'- `{rel}`：{path.read_text().strip()}')
for rel in [protocol.get('active_lr_selection'), protocol.get('active_lambda_selection')]:
    if not rel:
        continue
    path = ROOT/rel
    if path.exists():
        data = json.loads(path.read_text())
        values = {k: v for k,v in data.items() if k.startswith('selected_')}
        lines.append(f'- [{path.name}](../{rel})：`{json.dumps(values,ensure_ascii=False)}`')
lines += ['', '## 入口', '', '- [文献调研](../literature/文献调研.md)',
          '- [研究路线与时间安排](研究路线与时间安排.md)', '- [决策日志](决策日志.jsonl)',
          '- 大模型与完整checkpoint：`/tmp/grad-research-20261106`。',
          '- 本工作目录当前链接到临时盘；已有持久快照和主权重位于同级 `research_delivery/`。',
          '- [持久主权重验收](primary_weights_durable_publication.json)；[快照记录](durable_snapshot_latest.json)。', '']
(ROOT/'docs/STATUS.md').write_text('\n'.join(lines))
print(ROOT/'docs/STATUS.md')
