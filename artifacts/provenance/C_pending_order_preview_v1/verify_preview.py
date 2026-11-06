#!/usr/bin/env python3
"""Independent checks of immutable preview inputs and all eight FIFO projections."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from reconcile_main_behavior_runtime import fifo_projection
from schedule_jobs import digest_job

out = ROOT / 'results/C_pending_order_preview_v1'
x = json.loads((out / 'preview.json').read_text())
load = lambda name: json.loads((out / 'sources' / name).read_text())
sha = lambda blob: hashlib.sha256(blob).hexdigest()
raw, state, master = (load(name) for name in ('raw_runtime.json', 'master_state.json', 'master_queue.json'))
lookup = {job['id']: job for job in master}
model = {row['id']: tag for tag, update in raw['extended_model_runtime_updates'].items() for row in update['jobs']}
for name, proof in x['sources'].items():
    assert sha((out / 'sources' / name).read_bytes()) == proof['sha256']
assert sha((ROOT / 'configs/supplemental_v1_queue.json').read_bytes()) == x['sources']['master_queue.json']['sha256']
assert len(x['all564_definition_proofs']) == 564
for row in x['all564_definition_proofs']:
    assert digest_job(lookup[row['id']]) == row['definition_sha256'] == state['jobs'][row['id']]['definition_sha256']
original_ids = [job['id'] for job in master]
running = {jid for jid, record in state['jobs'].items() if record['status'] == 'running'}
c410 = {jid for jid, tag in model.items() if tag == 'pythia-410m'}
initial = {'id': 'timing_only_C410_extended_NLL_initial', 'depends_on': sorted(c410)}
phase = [initial] + [{'id': 'timing_only_C410_NLL_' + kind + '_' + jid, 'depends_on': [initial['id']]}
                   for kind in ('final', 'best') for jid in sorted(c410)]
rates = x['frozen_C410_NLL_high_rate_seconds']
durations = {job['id']: rates['initial' if job is initial else 'final' if '_NLL_final_' in job['id'] else 'best'] for job in phase}
checks = 0
for case in x['results']:
    ids = x['candidate_order_proofs'][case['candidate']]['all564_ordered_ids']
    assert len(ids) == len(set(ids)) == 564 and set(ids) == set(original_ids)
    for position, jid in enumerate(original_ids):
        if jid not in model or jid in running or model[jid] == 'pythia-410m' or state['jobs'][jid]['status'] != 'pending':
            assert ids[position] == jid
    for tag in set(model.values()):
        assert [jid for jid in ids if model.get(jid) == tag] == [jid for jid in original_ids if model.get(jid) == tag]
    candidate = [lookup[jid] for jid in ids]
    for v in case['variants']:
        early = v['variant'] == 'event_gated_C49_high'
        result = fifo_projection(candidate, state, phase if early else [], durations if early else {}, raw,
                                 raw['remaining_slot_hours'] * 3600, True)
        assert result == v['FIFO']
        assert len(v['all63_actual_or_forecast_training_starts']) == 63
        assert not ({row['id'] for row in result['future_dispatch_timing_rows']} & running)
        expected_ideal = (raw['remaining_slot_hours'] * 3600 + (sum(durations.values()) if early else 0)) * 1.1 / 4
        assert abs(result['ideal_four_slot_seconds_with_10pct'] - expected_ideal) < 1e-7
        if early:
            assert v['root_optional_new_early_work_both_time_gates'] == (
                v['minimum_launch_headroom_seconds'] >= 300 and v['finish_margin_before_2350_seconds'] >= 900)
        checks += 1
current = x['results'][0]['variants'][0]
last4 = x['results'][3]['variants'][0]
assert abs(last4['FIFO']['makespan_seconds_with_10pct'] - current['FIFO']['makespan_seconds_with_10pct']) < 1e-7
assert last4['minimum_launch_headroom_seconds'] > current['minimum_launch_headroom_seconds'] + 1000
assert [c['variants'][1]['root_optional_new_early_work_both_time_gates'] for c in x['results']] == [False, False, False, True]
receipt = {'schema_version': 1, 'status': 'passed', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
           'preview': {'path': str(out / 'preview.json'), 'sha256': sha((out / 'preview.json').read_bytes())},
           'checks': ['All source copies byte SHA exact', 'All564 definition hashes match source state',
                      'All4 candidates preserve running/nonC/C410 positions and within-model relative order',
                      'All8 FIFO projections independently recomputed from frozen snapshot',
                      'Whole FIFO10percent once including all49 high rates', 'All63 actual/forecast starts inspected',
                      'Last4-only candidate has unchanged makespan and >1000seconds more launch headroom',
                      'Optional49 threshold checks true only for last4 candidate'],
           'projections_recomputed': checks, 'live_master_queue_unchanged': True,
           'verifier': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).read_bytes())},
           'score_semantics_read': False, 'GPU_queue_or_scheduler_started': False}
target = out / 'verification_receipt.json'
with target.open('x') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'receipt': str(target), 'sha256': sha(target.read_bytes()), 'projections_checked': checks}))
