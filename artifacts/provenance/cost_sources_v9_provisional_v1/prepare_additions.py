#!/usr/bin/env python3
"""Prepare explicit small-metadata cost sources. Never run a cost collector."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CUT = '2026-11-06T19:34'


def read(path):
    path = Path(path).resolve()
    a = path.stat()
    if a.st_size > 2 << 20:
        raise ValueError('Only bounded small metadata/source reads: ' + str(path))
    raw = path.read_bytes()
    b = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if stamp(a) != stamp(b) or len(raw) != b.st_size:
        raise ValueError('Metadata changed during preparation')
    return raw, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def load(path):
    raw, proof = read(path)
    return json.loads(raw), proof


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def category(path):
    s = str(path).lower()
    if 'cache_hint' in s or '_hint_' in s:
        return 'storage_wait_and_reservation'
    if 'publication' in s:
        return 'storage_publication'
    if 'additional_endpoint_actual' in s:
        return 'lossless_export'
    if any(x in s for x in ('tree_restore', 'native_cpu_restore', 'native_completed', 'native_verification')):
        return 'archive_qa'
    return 'cpu_analysis'


def main():
    observed = dt.datetime.now(dt.timezone.utc).isoformat()
    base, base_proof = load(ROOT / 'configs/cost_inventory_final_sources_v8.json')
    completion, completion_proof = load(ROOT / 'artifacts/provenance/cost_sources_v8/COMPLETION.json')
    if completion['source_spec']['sha256'] != base_proof['sha256']:
        raise ValueError('Accepted v8 source differs')
    registered = set()
    for op in base['additional_operations']:
        canonical = op.get('canonical_id', '')
        if canonical.startswith('explicit_timer:'):
            registered.add(canonical.split('#')[0][len('explicit_timer:'):])
        source, _ = load(op['source']['path'])
        if isinstance(source, dict) and isinstance(source.get('original_source'), dict):
            registered.add(str(Path(source['original_source']['path']).resolve()))
    selected = {}
    excluded = []
    for path in sorted((ROOT / 'artifacts/provenance').rglob('*tim*.json')):
        s = str(path)
        if path.is_relative_to(OUT) or any(x in s for x in (
                '/retained', '/sources/', '/small_', '/test_end_to_end', '/source_copies/',
                '/exact_cost_sources/', '/failed_svg_render/', '/small_actual')):
            continue
        if path.stat().st_size > 65536:
            continue
        data, proof = load(path)
        if not isinstance(data, dict):
            continue
        dates = [v for k, v in data.items() if isinstance(v, str) and any(
            t in k.lower() for t in ('utc', 'created', 'finished', 'started', 'timestamp', 'generated', 'ended'))]
        if not any(v.startswith('2026-11-06') and CUT <= v[:16] for v in dates):
            continue
        wall = [k for k, v in data.items() if type(v) in (int, float) and 'wall' in k.lower() and 'seconds' in k.lower()]
        if len(wall) != 1 or type(data.get('returncode')) is not int:
            continue
        selected[str(path.resolve())] = (data, proof, '/' + wall[0])
    # Undated complete parent timers and two successful v8 invocations, which v8
    # explicitly did not register in itself. No arbitrary older-source backfill.
    for rel, pointer in [
        ('artifacts/provenance/cost_sources_v8/collection_whole_child_timing.json', '/whole_child_wall_seconds'),
        ('artifacts/provenance/cost_sources_v8/preparation_whole_child_timing.json', '/whole_child_wall_seconds'),
        ('results/primary_behavior_analysis_execution_v1/whole_process_timing.json', '/whole_child_process_wall_seconds'),
        ('artifacts/provenance/C_native_final_CPU_restore_actual_v1/1b/restore_whole_child_evidence/whole_child_timing.json', '/whole_child_process_wall_seconds'),
        ('artifacts/provenance/final_editable_tree_restore_independent_review_v2/whole_child_timing.json', '/whole_child_wall_seconds'),
    ]:
        p = ROOT / rel
        if p.exists():
            data, proof = load(p)
            selected[str(p.resolve())] = (data, proof, pointer)
    normalized_dir = OUT / 'normalized'
    normalized_dir.mkdir()
    operations, metadata, table = [], {}, []
    raw_hashes = set()
    for number, (path, (data, proof, wall_pointer)) in enumerate(sorted(selected.items()), 1):
        if path in registered or proof['sha256'] in raw_hashes:
            excluded.append({'source': proof, 'reason': 'already registered in v8 or identical timer copy'})
            continue
        raw_hashes.add(proof['sha256'])
        metadata[path] = proof
        wall = data[wall_pointer[1:]]
        if type(wall) not in (int, float) or wall < 0:
            raise ValueError('Invalid measured wall')
        cpu_keys = []
        for pair in [
            ('user_cpu_seconds', 'sys_cpu_seconds'), ('child_user_seconds', 'child_system_seconds'),
            ('whole_child_user_cpu_seconds', 'whole_child_sys_cpu_seconds'),
            ('user_seconds', 'system_seconds'),
        ]:
            if all(type(data.get(k)) in (int, float) for k in pair):
                cpu_keys = list(pair)
                break
        identity = 'v9_' + hashlib.sha256(path.encode()).hexdigest()[:16]
        scope = data.get('scope') or data.get('timing_scope') or data.get('accounting') or (
            'Actual whole child/parent timer only. Authoring/manual work unknown; adjacent internal stage timers are nested.')
        normalized = {'schema_version': 1, 'status': 'normalized_actual_measured_operation',
            'original_source': proof, 'original_wall_pointer': wall_pointer, 'wall_seconds': wall,
            'CPU_component_pointers': ['/' + k for k in cpu_keys],
            'CPU_components': {k: data[k] for k in cpu_keys},
            'recorded_inclusive_CPU_seconds': sum(data[k] for k in cpu_keys) if cpu_keys else None,
            'original_returncode': data['returncode'], 'scope': scope,
            'timing_semantics': 'independent_measured_complete_operation',
            'normalization': 'Copy measured wall; sum only the named user/system components. Failure rc retained. No calendar-derived elapsed.',
            'nested_subtimers_added': False, 'new_operation_created_by_normalization': False}
        target = normalized_dir / (identity + '.json')
        write(target, normalized)
        _, np = read(target)
        op = {'id': identity, 'category': category(path), 'source': np, 'status_pointer': '/status',
            'accepted_statuses': ['normalized_actual_measured_operation'], 'wall_pointer': '/wall_seconds',
            'process_cpu_pointer': '/recorded_inclusive_CPU_seconds',
            'timing_semantics': 'independent_measured_complete_operation',
            'add_to_recorded_wall_totals': True, 'parent_id': None,
            'canonical_id': 'explicit_timer:' + path + '#' + wall_pointer, 'scope': scope}
        operations.append(op)
        table.append({'id': identity, 'source_path': path, 'sha256': proof['sha256'],
            'category': op['category'], 'returncode': data['returncode'], 'wall_pointer': wall_pointer,
            'wall_seconds': wall, 'process_CPU_seconds': normalized['recorded_inclusive_CPU_seconds'],
            'classification': 'disjoint_measured_parent_candidate', 'additive_when_merged_once': True})
    # These are measured regions or complete administrative operations without an
    # external spawn-to-exit timer. Do not relabel them as full work parents.
    region_specs = [
        ('artifacts/provenance/final_evaluation_handoff_actual_v1/old_master_stop_receipt.json', '/operation_wall_seconds', 'storage_wait_and_reservation', 'Metadata/termination operation region; detached old scheduler lifetime is not this work.'),
        ('artifacts/provenance/A_behavior_presentation_visual_review_actual_v1/assembly_receipt.json', '/region_wall_seconds', 'cpu_analysis', 'PNG contact-sheet assembly only; manual visual review/authoring parent unknown.'),
        ('artifacts/provenance/C_native_final_CPU_restore_actual_v1/2p8b/ROOT_preexecution_resource_gate_attempt1.json', '/parent_wall_seconds', 'archive_qa', 'Measured inline resource-check region failed; no restore launched. Later attempt2 exact failed headroom/whole timer unknown.'),
    ]
    for rel, pointer, cat, scope in region_specs:
        p = ROOT / rel
        if not p.exists():
            continue
        data, proof = load(p)
        metadata[proof['path']] = proof
        identity = 'v9_region_' + hashlib.sha256(proof['path'].encode()).hexdigest()[:16]
        operations.append({'id': identity, 'category': cat, 'source': proof,
            'status_pointer': '/status', 'accepted_statuses': [data['status']], 'wall_pointer': pointer,
            'process_cpu_pointer': None, 'timing_semantics': 'partial_region_parent_unknown_not_additive',
            'add_to_recorded_wall_totals': False, 'parent_id': None,
            'canonical_id': 'explicit_timer:' + proof['path'] + '#' + pointer, 'scope': scope})
        table.append({'id': identity, 'source_path': proof['path'], 'sha256': proof['sha256'],
            'category': cat, 'returncode': None, 'wall_pointer': pointer,
            'wall_seconds': data[pointer[1:]], 'process_CPU_seconds': None,
            'classification': 'measured_region_not_complete_external_parent', 'additive_when_merged_once': False})
    unknowns = []
    for rel, explanation in [
        ('artifacts/provenance/C410_best_publication_actual_v1/outer_timing_receipt_failure.json', 'Publisher durable success has a known internal region already in v8. Outer wrapper rc1, whole wall/CPU/childrc unknown. Misplaced batch timer copy is not another publisher parent.'),
        ('artifacts/provenance/bounded_completed_verification_cache_hint_v2/preparation_attempt_failure_and_scope_note.json', 'Preassert failure untimed, exact fresh trigger headroom unrecorded. Following hint child is separately timed and does not establish necessity/causality.'),
        ('artifacts/provenance/C_native_final_CPU_restore_actual_v1/2p8b/ROOT_preexecution_resource_gate_attempt2_failure.json', 'Failed inline gate exact available memory and full timer unknown. Later observation is not the failed-gate value; no restore inferred.'),
        ('artifacts/provenance/C_native_final_CPU_restore_actual_v1/2p8b/ROOT_review_attempt1_failure.json', 'Failed metadata-only ROOT review, full manual parent unknown; not a restore.'),
        ('artifacts/provenance/additional_endpoint_actual_v4/ROOT_C1B_best_batch_attempt1_failure.json', 'Actual preauthority memory gate failed; no batch launched then. Full failed parent timer unknown.'),
        ('artifacts/provenance/C2p8B_analysis_preparation_2130_v1/ROOT_inline_review_attempt1_failure.json', 'Failed inline ROOT metadata review; complete parent timing unknown.'),
        ('artifacts/provenance/final_evaluation_handoff_actual_v1/ROOT_inline_review_attempt1_failure.json', 'Failed handoff metadata-review assertion retained; not a GPU evaluation and full timer unknown.'),
    ]:
        p = ROOT / rel
        if p.exists():
            _, proof = read(p)
            metadata[proof['path']] = proof
            unknowns.append({'source': proof, 'complete_parent_wall_seconds': None, 'additive': False, 'explanation': explanation})
    pending = [
        {'category': 'dynamic_training', 'action': 'Freeze final607 old-master plus final514 successor exact states and update the existing v1 source spec/merged-state inputs once. Do not add scheduler whole-slot timings as new operational work parents.', 'expected': '267 physical trainings incl9 D duplicates, 7.215B target predictions; frozen scope is not completion.'},
        {'category': 'dynamic_inference', 'action': 'Use all11 actual evaluation sources+3 prior plans and successor immutable queue/state, existing canonical inference+manifest SHA dedup and attestations. Final514=493forward+21reuse only a frozen scope, not yet completion.', 'unknown_pending': 'unfinished jobs, future result verification/scorer costs, final output closure'},
        {'category': 'C_final_dynamic_exports', 'action': 'Existing C_operations refreshes all63 draft records and3 independently timed assemblies from explicit root; prior first-created records must match immutable bytes. Add only actual new drafts/assemblies; daemon progress elapsed includes waiting and stays nonadditive.', 'avoid': 'Do not register draft/assembly a second time as static parents.'},
        {'category': 'C_final_publication', 'action': 'Actual1B/2.8B whole publisher timers are static candidates here. Do not also turn on C_exports publication_receipt as another additive internal timer for the same invocation.', 'avoid': 'Their docs internal publication wall is nested in outer whole.'},
        {'category': 'C_best_batches_and_publications', 'action': 'C1B/C2.8B sample parents are observed here. Full batch parent timers/completions/publication and metadata acceptance remain pending at observation; do not add per-endpoint/encode/decode/assembly subtimers on top of a whole batch parent.', 'expected_timer_roots': ['artifacts/provenance/additional_endpoint_actual_v4/C1B_best_batch', 'artifacts/provenance/additional_endpoint_actual_v4/C2p8B_best_batch']},
        {'category': 'native_CPU_restore', 'action': 'C1B original whole restore parent is observed. C2.8B actual restore, future bind/check/publication followups remain pending; failed gate is not a restore or zero cost.'},
        {'category': 'remaining_analysis', 'action': 'Future C1B/C2.8B NLL, C1B/C2.8B behavior full check/analyze, independent numerical reviews/presenters and ROOT report operations require actual new parent receipts. Existing primary/A/C410behavior timers only; do not extrapolate unseen scores or manual time.'},
        {'category': 'final_report_and_delivery', 'action': 'Final MD/deck/PDF render, visual review, project snapshot, editable-tree restore and final publication/acceptance require actual receipts; future failed attempts remain separate. Whole project elapsed/GPU telemetry remains a separate timewindow account.'},
        {'category': 'archived_QA', 'action': 'If ROOT later authorizes archived QA, copy its SMALL metadata/timers/queue/state/audit receipts into the worktree first, retaining external source path+SHA. Register QA training separately from267 scientific trainings, plus disjoint full-state postrun CPU parent. No model/archive payload copy required for cost sources.'},
        {'category': 'this_preparation_and_final_collection', 'action': 'This preparation whole-child timer is written by an external parent after the index and is not recursively registered here. Final v9 source preparation/collector/IO-audit and final reviews likewise require later explicit sources; do not pre-count.'},
    ]
    index = {'schema_version': 1, 'status': 'PROVISIONAL_ADDITIONS_ONLY_NOT_V9_COLLECTION',
        'observed_at_utc': observed, 'discovery_cutoff': CUT, 'preserved_v8': base_proof,
        'preserved_v8_completion': completion_proof, 'preserved_v8_inventory': completion['inventory'],
        'unchanged_collectors': [base['v1_collector'], base['unchanged_v2_collector'], base['revision8_unchanged_IO_audit_wrapper']],
        'collector_compatible_append_fragment': {'additional_operations': operations, 'metadata_sources': list(metadata.values())},
        'complete_measured_parent_candidates': sum(x['add_to_recorded_wall_totals'] for x in operations),
        'measured_failed_parent_candidates': sum(x['returncode'] != 0 for x in table if x['classification'] == 'disjoint_measured_parent_candidate'),
        'regions_not_additive': len(region_specs), 'excluded_already_registered_or_identical_copy': excluded,
        'manual_unknown_failures': unknowns, 'pending_actual_categories': pending,
        'no_totals_computed': True, 'collector_executed': False, 'v8_modified': False,
        'score_semantics_read': False, 'score_or_weight_or_archive_payload_bytes_read': 0,
        'scope_limitations': ['Source discovery reads only named small timer/receipt/source JSON, never score outputs or model/archive payloads.',
            'Operation parent wall is not GPU compute time or whole-study elapsed; independent parents may overlap in calendar time.',
            'User/system CPU components are summed only within one original measured parent; unavailable components remain null.',
            'Adjacent reviewer/scorer/body/encode/decode/SHA/stage timers are nested, not extra parents. No unrecorded work imputed as zero.',
            'Provisional list requires ROOT final source/identity review before merging with v8. Current metadata-only discovery does not authorize collection or any experiment.']}
    write(OUT / 'provisional_additions_index.json', index)
    with (OUT / 'provisional_operations.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({'status': index['status'], 'parent_candidates': index['complete_measured_parent_candidates'],
        'failed_parents': index['measured_failed_parent_candidates'], 'regions': index['regions_not_additive'],
        'unknown_failures': len(unknowns), 'no_totals_computed': True, 'collector_executed': False}))


if __name__ == '__main__':
    main()
