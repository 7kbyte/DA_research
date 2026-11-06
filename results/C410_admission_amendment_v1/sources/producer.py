#!/usr/bin/env python3
"""Read-only admission review of existing FIFO rows; no new simulation or jobs."""
import ast
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
PREVIEW = ROOT / 'results/C410_early_NLL_upperbound_v1/preview.json'
PREVIEW_SHA = '9d2b84c3d92aaa363690b5dcce010b4f6fb20fa2b30d147c9908b3b946d5c0ed'
SCHEDULER_SHA = 'c5c264b058ff655315ef896ccb0a89b471637c6612013a51999fc6f48b2d16d9'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def produce(output):
    if output.exists():
        raise FileExistsError('Use a new amendment namespace')
    blobs = {'preview.json': PREVIEW.read_bytes(),
             'scheduler_source.py': (ROOT / 'scripts/schedule_jobs.py').read_bytes(),
             'live_master_cmdline.bin': Path('/proc/2988506/cmdline').read_bytes(),
             'live_master_stat.txt': Path('/proc/2988506/stat').read_bytes(),
             'live_master_queue.json': (ROOT / 'configs/supplemental_v1_queue.json').read_bytes(),
             'producer.py': Path(__file__).read_bytes()}
    if digest(blobs['preview.json']) != PREVIEW_SHA or digest(blobs['scheduler_source.py']) != SCHEDULER_SHA:
        raise ValueError('Frozen preview/scheduler source changed')
    old = json.loads(blobs['preview.json'])
    for name in ['master_state.json', 'master_queue.json', 'raw_runtime.json']:
        blobs['preview_' + name] = (PREVIEW.parent / 'sources' / name).read_bytes()
        if digest(blobs['preview_' + name]) != old['source_proofs'][name]['sha256']:
            raise ValueError('Historical preview snapshot source changed')
    argv = [part.decode() for part in blobs['live_master_cmdline.bin'].split(b'\0') if part]
    if Path(argv[1]).resolve() != (ROOT / 'scripts/schedule_jobs.py').resolve():
        raise ValueError('Recorded live master is not the reviewed scheduler')
    defaults = {}
    for node in ast.walk(ast.parse(blobs['scheduler_source.py'])):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            if (node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in {'--launch-margin-seconds', '--stop-margin-seconds'}):
                for keyword in node.keywords:
                    if keyword.arg == 'default':
                        defaults[node.args[0].value] = ast.literal_eval(keyword.value)
    def option(flag, fallback=None):
        return argv[argv.index(flag) + 1] if flag in argv else fallback
    deadline = dt.datetime.fromisoformat(option('--deadline').replace('Z', '+00:00'))
    launch = float(option('--launch-margin-seconds', defaults['--launch-margin-seconds']))
    stop = float(option('--stop-margin-seconds', defaults['--stop-margin-seconds']))
    if deadline != dt.datetime.fromisoformat('2026-11-07T00:00:00+00:00') or launch != 1800 or stop != 600:
        raise ValueError('Actual reviewed master policy changed; do a new review')
    live_queue = json.loads(blobs['live_master_queue.json'])
    preview_queue = json.loads(blobs['preview_master_queue.json'])
    if live_queue != preview_queue:
        raise ValueError('Current master definitions/order differ from preview source')
    jobs = {job['id']: job for job in preview_queue}
    state = json.loads(blobs['preview_master_state.json'])
    raw = json.loads(blobs['preview_raw_runtime.json'])
    c_models = {row['id']: model for model, update in raw['extended_model_runtime_updates'].items() for row in update['jobs']}
    if len(c_models) != 63 or {jobs[identity]['expected_seconds'] for identity in c_models} != {3600}:
        raise ValueError('Require all63 unchanged C definitions with3600-second allowances')
    when = dt.datetime.fromisoformat(old['runtime_source_utc'])
    cases = []
    forecasts = [('no_early_NLL', old['training_only_FIFO'])] + [(case['name'], case['FIFO']) for case in old['scenarios']]
    for name, projection in forecasts:
        forecast_rows = {row['id']: row for row in projection['future_dispatch_timing_rows'] if row['id'] in c_models}
        rows = []
        for identity, model in c_models.items():
            record = state['jobs'][identity]
            forecast = identity in forecast_rows
            start = when + dt.timedelta(seconds=forecast_rows[identity]['start_seconds']) if forecast else dt.datetime.fromisoformat(record['started_at_utc'])
            cutoff = deadline - dt.timedelta(seconds=launch + jobs[identity]['expected_seconds'])
            headroom = (cutoff - start).total_seconds()
            rows.append({'id': identity, 'model': model, 'basis': 'forecast_dispatch_row' if forecast else 'already_started_actual_record',
                         'preview_state_status': record['status'], 'expected_seconds': jobs[identity]['expected_seconds'],
                         'start_utc': start.isoformat(), 'strict_launch_cutoff_utc': cutoff.isoformat(),
                         'launch_headroom_seconds': headroom, 'satisfies_existing_strict_admission': headroom > 0})
        assert len(rows) == 63
        violations = [row for row in rows if not row['satisfies_existing_strict_admission']]
        future = [row for row in rows if row['basis'] == 'forecast_dispatch_row']
        latest = max(future, key=lambda row: row['start_utc'])
        cases.append({'name': name, 'C_training_launches_reviewed': 63, 'already_started_actual_records': len(rows) - len(future),
                      'future_C_launch_rows_reviewed': len(future), 'latest_forecast_C_launch': latest,
                      'forecast_admission_violation_count': len(violations), 'forecast_admission_violations': violations,
                      'all63_satisfy_actual_master_launch_policy_under_old_timing_scenario': not violations,
                      'all_launch_rows': rows})
    result = {'schema_version': 1, 'status': 'admission_omission_amendment_not_execution_authority',
        'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'historical_preview': {'path': str(PREVIEW), 'sha256': PREVIEW_SHA},
        'historical_preview_original_bytes_preserved': True, 'historical_preview_runtime_source_utc': old['runtime_source_utc'],
        'live_master': {'pid': 2988506, 'argv': argv, 'deadline_utc': deadline.isoformat(),
            'launch_margin_seconds': launch, 'launch_margin_origin': 'actual argv override' if '--launch-margin-seconds' in argv else 'reviewed argparse default',
            'stop_margin_seconds': stop, 'stop_margin_origin': 'actual argv override' if '--stop-margin-seconds' in argv else 'reviewed argparse default',
            'nominal_stop_new_and_running_threshold_utc': (deadline - dt.timedelta(seconds=stop)).isoformat()},
        'scheduler_source': {'path': str(ROOT / 'scripts/schedule_jobs.py'), 'sha256': SCHEDULER_SHA,
            'default_cli_lines': [83, 84], 'admission_check_line': 159,
            'actual_reject_expression': 'now + expected_seconds >= deadline - launch_margin_seconds'},
        'all63_expected_seconds': 3600, 'strict_training_launch_cutoff_utc': '2026-11-06T22:30:00+00:00',
        'cases': cases,
        'conclusion': 'Old early-C49 low/central/high finish-margin checks are insufficient: all three old timing scenarios violate actual launch admission for pending C training.',
        'future_gate': ['At actual complete-phase READY, recapture actual manifest/queue/master argv/state/runtime.',
            'Review every one of all63 training starts against its actual strict admission cutoff, with explicit operating headroom beyond polling/start overhead.',
            'Independently retain >=15min finish reserve before23:50, all horizons/cohorts, and the full30min final delivery tail.',
            'Do not multiply the already included sensitivity factors or10 percent again; a later actual-time recalculation can have less remaining uncertainty.',
            'Do not pre-assume a restart, changed job allowance, changed deadline, changed margins, or permission to alter any of them.'],
        'scope_limits': ['These are old10-percent scenario forecasts, not observed failed admissions or proof that future actual runs will fail.',
            'Even no-early scenario has only33.125s forecast launch headroom at its last job; positivity alone is not a robust guarantee.',
            'B has already actually completed; its attested history is not rewritten. Earlier B/main66 predictions also did not explicitly review this admission gate.',
            'No source, queue, scheduler, authority, horizon, checkpoint, score or global budget is changed by this amendment.'],
        'sources': {name: {'sha256': digest(payload), 'bytes': len(payload)} for name, payload in blobs.items()},
        'new_GPU_queue_authority_watcher_or_framework': False, 'score_semantics_read': False}
    output.mkdir()
    sources = output / 'sources'
    sources.mkdir()
    for name, payload in blobs.items():
        write(sources / name, payload)
    write(output / 'amendment.json', (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())
    with (output / 'launch_admission.csv').open('x', newline='') as stream:
        fields = ['scenario', 'id', 'model', 'basis', 'start_utc', 'strict_launch_cutoff_utc', 'launch_headroom_seconds', 'satisfies_existing_strict_admission']
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for case in cases:
            for row in case['all_launch_rows']:
                writer.writerow({'scenario': case['name'], **{key: row[key] for key in fields if key != 'scenario'}})
    print(json.dumps({'path': str(output / 'amendment.json'), 'sha256': digest((output / 'amendment.json').read_bytes()),
        'violations': {case['name']: case['forecast_admission_violation_count'] for case in cases}}))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: producer.py NEW_AMENDMENT_DIRECTORY')
    produce(Path(sys.argv[1]).resolve())
