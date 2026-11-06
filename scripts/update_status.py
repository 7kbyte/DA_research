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
lines = ['# 研究运行状态', '', f'更新时间：{now.isoformat()}',
         '截止时间：2026-11-07 03:26:32 UTC（北京时间11:26）',
         f'已用 {(now-started).total_seconds()/3600:.2f} 小时；剩余 {max(0,(deadline-now).total_seconds()/3600):.2f} 小时。',
         '', '以下状态来自运行日志。配置完成不等于实验完成；正式结论以最终研究报告为准。',
         '', '| 调度阶段 | 完成 | 运行 | 待执行 | 失败/阻塞 |', '|---|---:|---:|---:|---:|']
for path in sorted((ROOT/'logs').glob('*job_state.json')):
    try:
        state = json.loads(path.read_text())
        c = collections.Counter(x['status'] for x in state['jobs'].values())
        lines.append(f"| {path.stem} | {c['completed']} | {c['running']} | {c['pending']} | {c['failed']+c['blocked']} |")
    except (ValueError, KeyError):
        lines.append(f'| {path.stem} | 状态暂不可读 | | | |')
lines += ['', '## 当前数据与选参记录', '']
for rel in ['artifacts/data/VERSION', 'artifacts/evaluation_v3/VERSION']:
    path = ROOT/rel
    if path.exists():
        lines.append(f'- `{rel}`：{path.read_text().strip()}')
for rel in ['results/pilot_lr_selection_v3.json', 'results/pilot_lambda_selection_v3.json']:
    path = ROOT/rel
    if path.exists():
        data = json.loads(path.read_text())
        values = {k: v for k,v in data.items() if k.startswith('selected_')}
        lines.append(f'- [{path.name}](../{rel})：`{json.dumps(values,ensure_ascii=False)}`')
lines += ['', '## 入口', '', '- [文献调研](../literature/文献调研.md)',
          '- [研究路线与时间安排](研究路线与时间安排.md)', '- [决策日志](决策日志.jsonl)',
          '- 大模型与完整checkpoint：`/tmp/grad-research-20261106`。',
          '- 数据、代码、结果、日志与文档保存在本工作目录；最终将注明关键权重的持久化位置。', '']
(ROOT/'docs/STATUS.md').write_text('\n'.join(lines))
print(ROOT/'docs/STATUS.md')
