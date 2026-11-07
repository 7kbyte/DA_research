#!/usr/bin/env python3
"""One CPU-only fixed two-slot export estimate; no launch or reservation."""
import datetime as dt
import hashlib
import heapq
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/optional_weight_expansion_planning_v1'
GI = 1 << 30
UTC = dt.timezone.utc
parse = lambda x: dt.datetime.fromisoformat(x.replace('Z', '+00:00'))
sources = {}
blobs = {}


def read(name, path):
    path = Path(path).resolve(); before = path.stat(); data = path.read_bytes(); after = path.stat()
    assert len(data) < 32 << 20 and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    sources[name] = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}; blobs[name] = data
    return json.loads(data)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    now = dt.datetime.now(UTC)
    a = read('A_completion.json', ROOT / 'artifacts/assets/A_selected_weight_export/completion.json')
    assert a['status'] == 'passed' and a['new_unique_weights'] == 15
    a_rate = a['wall_seconds'] / a['new_unique_weights']
    diagnostic = read('C_first_parent_review.json', ROOT / 'docs/C首例权重资源独立复核.json')
    rates = {}; native = {}
    for model, item in diagnostic['models'].items():
        record = read(model + '_full_final_parent.json', item['source_record']['path'])
        assert sources[model + '_full_final_parent.json']['sha256'] == item['source_record']['sha256']
        assert record['additional_encoding_calls'] == 1 and record['bytewise_verification']['all_tensor_bytes_equal']
        rates[model] = record['wall_seconds']
        base = Path('/tmp/grad-research-20261106/models') / (model + '-final')
        native[model] = {'bytes': sum(p.stat().st_size for p in base.iterdir() if p.is_file()),
                         'scope': 'Native base/config/tokenizer/README file sizes only; no payload read.'}
    primary = read('primary_two_package_state.json', ROOT / 'results/primary_remaining_endpoints_export_state.json')
    tar_rates = []
    for index, tag in enumerate(('budget_final_weights', 'primary_best_weights')):
        before = read(tag + '_before_tar.json', Path('/tmp/grad-research-20261106/delivery') / (tag + '_manifest.json'))
        finish = primary['completed_packages'][index]
        seconds = finish['wall_seconds'] - before['wall_seconds_before_tar']
        assert seconds > 0
        tar_rates.append({'package': tag, 'bytes': finish['tar_bytes'], 'stage_wall_seconds': seconds,
                          'bytes_per_second': finish['tar_bytes'] / seconds,
                          'scope': 'Difference of same-start cumulative prefixes: manifest-before-TAR through package completion, including build/member/full-TAR verification/fsync. Prefixes are not summed.'})
    tar_rate = min(row['bytes_per_second'] for row in tar_rates)
    publications = []
    for tag in ('A_selected_weights', 'primary_best_weights', 'budget_final_weights'):
        pub = read(tag + '_publication.json', ROOT / 'docs' / (tag + '_durable_publication.json'))
        assert pub['status'] == 'passed'
        publications.append({'package': tag, 'bytes': pub['bytes'], 'wall_seconds': pub['wall_seconds'],
                             'bytes_per_second': pub['bytes'] / pub['wall_seconds']})
    pub_rate = min(row['bytes_per_second'] for row in publications)
    preview = read('fixed_order_preview.json', ROOT / 'results/C_pending_order_preview_v1/preview.json')
    assert sources['fixed_order_preview.json']['sha256'] == '24b25d5ba2418f825ee441aa08c22a2514d102a7d5008e1bd77a2ac577283502'
    chosen = next(row for row in preview['results'] if row['candidate'] == 'last4_1b_after_2p8b')
    reservations = read('ROOT_capacity_sequence.json', ROOT / 'artifacts/provenance/extra_endpoint_reservations_v1/sequence.json')
    assert sources['ROOT_capacity_sequence.json']['sha256'] == '8fe366e175b789494ea606377ab9393868747672a5262293f503acf91ff08e27'
    allocation_observations = []
    for item in reservations['entries']:
        spec = read(item['name'] + '_capacity_plan.json', item['plan'])
        assert sources[item['name'] + '_capacity_plan.json']['sha256'] == item['plan_sha256']
        stat = Path(spec['path']).stat()
        assert stat.st_dev == spec['identity']['device'] and stat.st_ino == spec['identity']['inode']
        allocation_observations.append({'name': item['name'], 'target_bytes': spec['target_bytes'],
            'actual_allocated_bytes_at_read_only_stat': stat.st_blocks*512, 'current_size_at_stat': stat.st_size,
            'initial_bytes_bound_by_plan': spec['initial_allocated_bytes'], 'writer_may_advance_after_observation': True})
    jobs = [('A_remaining_final', 0, 'pythia-410m', 93, 115), ('C410_best', 0, 'pythia-410m', 24, 30),
            ('C1B_best', 0, 'pythia-1b', 24, 75), ('B_final', 1, 'pythia-410m', 60, 75),
            ('C2p8B_best', 1, 'pythia-2.8b', 15, 140)]
    definitions = {tag: {'slot': slot, 'model': model, 'no_alias_unique_count': count,
        'conditional_D_alias_unique_count': 21 if tag in ('C410_best', 'C1B_best') else count,
        'ceiling_GiB': ceiling, 'parent_roundtrip_seconds_proxy': a_rate if tag in ('A_remaining_final', 'B_final') else rates[model],
        'assembly_seconds_proxy': (ceiling * GI + native[model]['bytes']) / tar_rate,
        'publication_seconds_proxy': ceiling * GI / pub_rate} for tag, slot, model, count, ceiling in jobs}
    cutoff = parse('2026-11-07T02:00:00Z'); tail = parse('2026-11-07T02:56:32Z')
    cases = []
    for variant in chosen['variants']:
        release = {tag: parse(value) for tag, value in variant['model_complete_phase_times_utc_proxy'].items()}
        for alias in (False, True):
            for factor in (1., 1.25):
                for ready_reserve in (0, 1800):
                    chains = [['A_remaining_final', 'C410_best', 'C1B_best'], ['B_final', 'C2p8B_best']]
                    heap = []; planned = {}; pub_available = now
                    def begin(slot, index, available):
                        identity = chains[slot][index]; item = definitions[identity]
                        count = item['conditional_D_alias_unique_count'] if alias else item['no_alias_unique_count']
                        eligible = now if identity in ('A_remaining_final', 'B_final') else release[item['model']] + dt.timedelta(seconds=ready_reserve)
                        start = max(available, eligible)
                        cost = item['parent_roundtrip_seconds_proxy'] * factor
                        record = {'id': identity, 'slot': slot, 'unique_encodings': count, 'logical_endpoints': item['no_alias_unique_count'],
                            'earliest_release_used_utc': eligible.isoformat(), 'first_encoding_start_utc_proxy': start.isoformat(),
                            'last_encoding_start_utc_proxy': (start + dt.timedelta(seconds=(count-1)*cost)).isoformat(),
                            'roundtrip_wall_seconds_proxy': count * cost,
                            'assembly_wall_seconds_proxy': item['assembly_seconds_proxy'] * factor,
                            'publication_wall_seconds_proxy': item['publication_seconds_proxy'] * factor}
                        before_pub = start + dt.timedelta(seconds=count*cost + item['assembly_seconds_proxy']*factor)
                        planned[identity] = record
                        heapq.heappush(heap, (before_pub, slot, index, identity))
                    begin(0, 0, now); begin(1, 0, now)
                    while heap:
                        request, slot, index, identity = heapq.heappop(heap); row = planned[identity]
                        pub_start = max(request, pub_available)
                        finish = pub_start + dt.timedelta(seconds=row['publication_wall_seconds_proxy'])
                        pub_available = finish
                        row.update({'publication_serial_wait_seconds_proxy': (pub_start-request).total_seconds(),
                            'package_durable_finish_utc_proxy': finish.isoformat(),
                            'all_new_encodings_strictly_before0200': parse(row['last_encoding_start_utc_proxy']) < cutoff,
                            'package_finishes_before30minute_delivery_tail': finish <= tail,
                            'tail_margin_seconds_proxy': (tail-finish).total_seconds()})
                        if index + 1 < len(chains[slot]): begin(slot, index+1, finish)
                    cases.append({'training_order_variant': variant['variant'], 'alias_21_21_requires_actual_best_proofs': alias,
                        'new_CPU_work_wall_factor': factor, 'additional_C_READY_delay_reserve_seconds': ready_reserve,
                        'zero_ready_delay_case_is_optimistic_lower_bound_not_permission': ready_reserve == 0,
                        'rows': list(planned.values()), 'all_five_time_gates_pass_under_assumptions': all(row['all_new_encodings_strictly_before0200'] and row['package_finishes_before30minute_delivery_tail'] for row in planned.values())})
    latest = []
    for variant in chosen['variants']:
        for alias in (False, True):
            for identity in ('C410_best', 'C1B_best', 'C2p8B_best'):
                item = definitions[identity]; count = item['conditional_D_alias_unique_count'] if alias else item['no_alias_unique_count']
                factor = 1.25; per = item['parent_roundtrip_seconds_proxy'] * factor
                by_cutoff = cutoff - dt.timedelta(seconds=(count-1)*per)
                by_tail = tail - dt.timedelta(seconds=(count*item['parent_roundtrip_seconds_proxy'] + item['assembly_seconds_proxy'] + item['publication_seconds_proxy'])*factor)
                bound = min(by_cutoff, by_tail); training_end = parse(variant['model_complete_phase_times_utc_proxy'][item['model']])
                latest.append({'variant': variant['variant'], 'id': identity, 'conditional_aliases': alias,
                    'latest_first_start_exclusive_or_tail_bound_utc_proxy': bound.isoformat(),
                    'extra_READY_delay_isolated_upper_headroom_seconds': (bound-training_end).total_seconds(),
                    'scope': 'Isolated high1.25 bound before other-slot publication waiting/existing Cfinal priority. Actual READY and current full-chain budget must be rechecked.'})
    shared = os.statvfs('/data/202610/7kb/research_delivery'); available = shared.f_bavail * shared.f_frsize
    ceiling = sum(item['ceiling_GiB'] for item in definitions.values()) * GI
    result = {'schema_version': 1, 'status': 'CPU_planning_only_not_launch_or_capacity_authority', 'captured_at_utc': now.isoformat(),
        'sources': sources, 'producer': {'path': str(Path(__file__).resolve()), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'capacity': {'five_package_ceiling_bytes': ceiling, 'five_package_ceiling_GiB': ceiling/GI,
            'ROOT_reported_available_bytes': 893669208064, 'ROOT_reference_remaining_after_five_ceilings_bytes': 893669208064-ceiling,
            'actual_sequence_before_initial_available_bytes': reservations['shared_available_bytes_before_initial_allocation'],
            'remaining_after_full_five_targets_from_sequence_before_initial_bytes': reservations['shared_available_bytes_before_initial_allocation']-ceiling,
            'current_statvfs_available_bytes_after_writer_started_not_charged_full435again': available,
            'actual_initial_five_bytes_bound_by_sequence': sum(row['initial_bytes_bound_by_plan'] for row in allocation_observations),
            'writer_capacity_observations_nonatomic_and_not_fulltarget_acceptance': allocation_observations,
            'existing_reservations_unchanged': True, 'this_planner_did_not_reserve_capacity': True,
            'actual_ROOT_writer_active_full_targets_still_require_receipt_and_ROOT_acceptance': True,
            'ceilings_are_enforced_engineering_caps_not_proven_best_compression_upper_bounds': True},
        'fixed_two_slot_chains': [['A_remaining_final', 'C410_best', 'C1B_best'], ['B_final', 'C2p8B_best']],
        'maximum_encoding_workers_including_existing_Cfinal': 3, 'CPU_threads_per_worker': 2,
        'A_15_full_parent_seconds': a['wall_seconds'], 'A_B_per_weight_inclusive_parent_proxy': a_rate,
        'C_full_final_parent_proxy_seconds_not_best_measurements': rates, 'native_base_bytes_stat_only': native,
        'assembly_rate_evidence': tar_rates, 'selected_slow_measured_assembly_bytes_per_second': tar_rate,
        'publication_rate_evidence': publications, 'selected_slow_measured_publication_bytes_per_second': pub_rate,
        'assembly_allowance_rule': 'Ceiling TAR bytes plus one extra native-base-size pass, divided by slower measured TAR-stage throughput. Additional proof/metadata READY latency remains unmeasured. No ceiling reduction from aliases.',
        'A_B_proxy_overlap': 'A1527.69 parent already contains assembly; extra assembly allowance here is intentional conservative overlap in a forecast, not additive measured cost.',
        'publication_lane_is_serial_in_new_five_package_estimate': True,
        'unmeasured_existing_Cfinal_IO_contention_and_prioritized_publication_not_a_measured_zero': True,
        'training_preview_already_contains_once10pct_no_additional_training_factor': True,
        'actual_complete_READY_audit_proof_latency_is_unknown_not_zero': True,
        'new_slots_hypothetical_start_at_capture_not_actual_launch': True,
        'definitions': definitions, 'cases': cases, 'latest_C_ready_bounds_high1p25': latest,
        'new_encoding_start_cutoff_utc_exclusive': cutoff.isoformat(), 'final_delivery_tail_starts_utc': tail.isoformat(),
        'reject_before_optional_phase_if_actual_READY_or_bound_or_capacity_missing': True,
        'in_progress_operation_may_finish_after0200_but_no_new_record_starts_then': True,
        'scientific_training_evaluation_original_Cfinal_and_delivery_take_priority': True,
        'score_semantics_or_large_payload_read': False, 'reservation_export_GPU_ordering_or_authority_started_or_changed': False}
    OUT.mkdir(); (OUT/'sources').mkdir()
    for name, data in blobs.items(): (OUT/'sources'/name).write_bytes(data)
    write(OUT/'plan.json', result)
    print(json.dumps({'path': str(OUT/'plan.json'), 'sha256': hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest(),
        'case_count': len(cases), 'capacity': result['capacity'], 'high_noalias_C2p8_rows': [row for case in cases if not case['alias_21_21_requires_actual_best_proofs'] and case['new_CPU_work_wall_factor']==1.25 for row in case['rows'] if row['id']=='C2p8B_best']}, ensure_ascii=False))


if __name__ == '__main__': main()
