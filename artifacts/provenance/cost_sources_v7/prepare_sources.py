"""One explicit v7 cost-source revision; unchanged collectors retain their semantics."""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def proof(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(source):
    path = Path(source['path'])
    assert path.suffix == '.json' and path.stat().st_size <= 32 << 20
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source['sha256']
    return json.loads(raw)


def pointer(value, path):
    for key in path.strip('/').split('/'):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def item(identity, category, path, wall, cpus, scope, rc=None, **extra):
    return dict(id=identity, category=category, source=proof(path), wall_pointer=wall,
                CPU_component_pointers=cpus, scope=scope, returncode_pointer=rc, **extra)


def main():
    previous = proof('configs/cost_inventory_final_sources_v6.json')
    assert previous['sha256'] == 'e4c99db9b1803681ad8706bb90480ede0be584975324040df979c1c9966b17e0'
    collector = proof('scripts/inventory_study_costs_final_v2.py')
    assert collector['sha256'] == '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae'
    spec = read(previous)
    candidate_source = proof('artifacts/provenance/cost_sources_v7_candidates_v1/candidates.json')
    candidates = read(candidate_source)
    parents = copy.deepcopy(candidates['eligible_disjoint_measured_parent_operations'])
    assert len(parents) == 18 and not any(v['already_registered_v6_operation_ids'] for v in parents)
    whole = '/whole_child_wall_seconds'
    ordinary_cpu = ['/user_cpu_seconds', '/sys_cpu_seconds']
    native = 'artifacts/provenance/C_native_final_restore_preparation_v1/'
    parents += [
        item('extra_export_B_all_final_batch_whole_child', 'lossless_export',
             'artifacts/provenance/additional_endpoint_actual_v4/B_all_final_batch/whole_child_timing.json',
             '/wall_seconds', ordinary_cpu,
             'Actual completed exporter child: startup/resume,59 new endpoints and assembly. First max1 sample was a separate v6 operation; references to it do not encode it again.', rc='/returncode'),
        item('B_all_final_publication_metadata_review_whole_parent', 'storage_publication',
             'artifacts/provenance/B_all_final_publication_metadata_review_actual_v1/whole_child_timing.json',
             '/whole_parent_wall_seconds', ['/parent_self_user_seconds','/parent_self_system_seconds','/children_user_seconds','/children_system_seconds'],
             'Inclusive parent around actual metadata review child; child1.811s and reviewer body1.5686s are nested.', rc='/returncode'),
        item('B_all_final_publication_whole_child', 'storage_publication',
             'artifacts/provenance/B_all_final_publication_actual_v1/whole_child_timing.json',
             '/wall_seconds', ordinary_cpu, 'Actual whole publisher child; internal copy/hash/readback/fsync/publication timer is nested.', rc='/returncode'),
        item('defense_workdraft_render_attempt01', 'cpu_analysis',
             'presentations/答辩工作稿_20261106_1730/render_attempts/attempt01/whole_child_timing.json',
             whole, ['/child_user_seconds','/child_system_seconds'],
             'Independent render child including charts/HTML/PDF/raster; excludes interactive authoring/visual review. Parent-self CPU is outside this measured child.', rc='/returncode'),
        item('defense_workdraft_render_attempt02', 'cpu_analysis',
             'presentations/答辩工作稿_20261106_1730/render_attempts/attempt02/whole_child_timing.json',
             whole, ['/child_user_seconds','/child_system_seconds'],
             'Second independent actual render child; first attempt is not nested or overwritten.', rc='/returncode'),
        item('CPU_full840_test_suite_whole_child', 'cpu_analysis', 'logs/tests_after_behavior_native_restore_v1_full.json',
             whole, ordinary_cpu, 'Whole840 tests child; testpaths=tests excludes provenance native29 and timeout6. Other targeted/full runs are separate actual executions.', rc='/returncode'),
        item('evaluation_budget_v6_refresh_1744_whole_child', 'cpu_analysis',
             'artifacts/provenance/evaluation_budget_v6_refresh_1745_v1/whole_child_timing.json', whole, ordinary_cpu,
             'Separate actual score-blind budget refresh child ending17:44:32; internal forecast body is nested.', rc='/returncode'),
        item('native_restore_author_28_development_tests', 'archive_qa', native+'attempt1_28_synthetic_tests_whole_child_timing.json',
             whole, ['/whole_child_user_seconds','/whole_child_sys_seconds'], 'Separate actually measured development28-test child; no real native package restore.', rc='/returncode'),
        item('native_restore_author_initial29_tests', 'archive_qa', native+'synthetic_tests_whole_child_timing.json',
             whole, ['/whole_child_user_seconds','/whole_child_sys_seconds'], 'Initial29-test whole child before timeout repair; preserved, not included in840.', rc='/returncode'),
        item('native_restore_author_timeout6_targeted_tests', 'archive_qa', native+'timeout_repair_tests_whole_child_timing.json',
             whole, ['/whole_child_user_seconds','/whole_child_sys_seconds'], 'Whole6-targeted-test child including both process-group toy subprocesses; copied inner toy receipts are not new charges.', rc='/returncode'),
        item('native_restore_ROOT_post_repair_original29_tests', 'archive_qa', native+'post_repair_original29_ROOT_whole_child_timing.json',
             whole, ordinary_cpu, 'Separate ROOT post-repair29-test execution, outside840 and previously completed timeout6.', rc='/returncode'),
        item('native_restore_independent_original_failure_probe', 'archive_qa',
             'artifacts/provenance/C_native_final_restore_independent_review_v1/timeout_probe.json',
             '/whole_probe_parent_timing/wall_seconds', ['/whole_probe_parent_timing/user_cpu_seconds','/whole_probe_parent_timing/system_cpu_seconds'],
             'Independent original timeout-bug reproduction+cleanup parent. Inner wrapper1.008s is nested; no real payload.'),
        item('native_restore_independent_fixed_probe', 'archive_qa',
             'artifacts/provenance/C_native_final_restore_independent_review_v1/timeout_fix_probe.json',
             '/whole_probe_parent_timing/wall_seconds', ['/whole_probe_parent_timing/user_cpu_seconds','/whole_probe_parent_timing/system_cpu_seconds'],
             'Independent fixed toy+admission/D-state review parent. COMPLETE repeats this same timer; do not charge it again.'),
        item('behavior_independent_review_v1_26tests', 'cpu_analysis',
             'artifacts/provenance/behavioral_independent_review_preparation_v1/tests_whole_child_timing.json',
             '/wall_seconds', ['/user_cpu_seconds','/system_cpu_seconds'],
             'Independent26-test child including synthetic CLI subtimers, all nested. No real behavioral score parsed.'),
        item('behavior_independent_review_v2_11tests', 'cpu_analysis',
             'artifacts/provenance/behavioral_independent_review_preparation_v2/tests_whole_child_timing.json',
             '/wall_seconds', ['/user_cpu_seconds','/system_cpu_seconds'],
             'Independent11-source-branch-test child; distinct from v1 and whole840 invocation.'),
        item('cost_v7_initial_failed_collection_whole_child', 'cpu_analysis',
             'artifacts/provenance/cost_sources_v7_failed_attempt1/collection_whole_child_timing.json',
             whole, ['/whole_child_user_seconds','/whole_child_sys_seconds'],
             'Actual failed first collection child: unsupported new category labels, no completed inventory. Separate from final successful collection and all historical cost operations.',
             rc='/returncode',observed_returncode=1),
    ]
    assert len(parents) == 34
    regions = copy.deepcopy(candidates['separate_partial_regions_not_in_parent_subtotal'])
    regions += [
        item('B_all_final_independent_publication_metadata_region', 'cpu_analysis',
             'docs/B_all_final_durable_publication_independent_review_v1.json', '/timing/wall_seconds_until_review_receipt_serialization',
             ['/timing/self_user_seconds','/timing/self_system_seconds'], 'Independent small metadata/stat review region, excluding unmeasured full authoring parent; not another publisher.'),
        item('delivery_guide_v4_source_count_region', 'cpu_analysis', 'docs/交付文件与恢复容量指南_来源核对_v4.json',
             '/timing/wall_seconds_until_receipt_serialization', ['/timing/self_user_seconds','/timing/self_system_seconds'],
             'Small fresh six-manifest counting/document region; full engineering parent unknown, no archive payload read.'),
        item('cost_v7_candidate_metadata_preparation_region', 'cpu_analysis', candidate_source['path'],
             '/candidate_preparation_region_wall_seconds', ['/candidate_preparation_region_process_CPU_seconds'],
             'Previously measured candidate metadata preparation region only; not whole candidate authorship.'),
    ]
    category_mapping={'restoration_QA':'archive_qa','report_rendering':'cpu_analysis','report_QA':'cpu_analysis',
        'engineering_tests':'cpu_analysis','lossless_project_archive':'lossless_export','archive_QA':'archive_qa',
        'cpu_analysis_prepare':'cpu_analysis'}
    for entry in parents+regions:
        entry['candidate_category']=entry.get('category','cpu_analysis')
        entry['category']=category_mapping.get(entry['candidate_category'],entry['candidate_category'])
        assert entry['category'] in {'cpu_analysis','opaque_prehash','lossless_export','storage_publication','archive_qa','storage_wait_and_reservation'}
    registered_paths = {Path(v['source']['path']).resolve() for v in spec['additional_operations']}
    new_rows = []; normalized = []; original_metadata = {}; summary = []
    def normalize(entry, is_parent):
        source = entry['source']; value = read(source)
        original_metadata[str(Path(source['path']).resolve())] = source
        wall = pointer(value, entry['wall_pointer'])
        cpus = entry.get('CPU_component_pointers') or ([entry['process_cpu_pointer']] if entry.get('process_cpu_pointer') else [])
        components = {p:pointer(value,p) for p in cpus}
        assert isinstance(wall,(int,float)) and not isinstance(wall,bool) and wall >= 0
        assert all(isinstance(v,(int,float)) and not isinstance(v,bool) and v>=0 for v in components.values())
        rc = pointer(value,entry['returncode_pointer']) if entry.get('returncode_pointer') else None
        if entry.get('returncode_pointer'):
            assert type(rc) is int
            if 'observed_returncode' in entry:
                assert rc == entry['observed_returncode']
            else:
                assert rc == 0
        if 'measured_wall_seconds' in entry:
            assert wall == entry['measured_wall_seconds']
        if is_parent:
            assert Path(source['path']).resolve() not in registered_paths, 'Explicit source already registered by v6'
        identity = entry['id']; path = HERE/(identity+'_normalized.json')
        semantics = 'independent_measured_complete_operation' if is_parent else 'partial_region_parent_unknown_not_additive'
        norm = dict(schema_version=1,status='normalized_actual_measured_operation',original_source=source,
                    original_wall_pointer=entry['wall_pointer'],wall_seconds=wall,CPU_component_pointers=cpus,
                    CPU_components=components,recorded_inclusive_CPU_seconds=sum(components.values()) if components else None,
                    original_returncode=rc,scope=entry['scope'],timing_semantics=semantics,
                    normalization='Measured wall copied; only named disjoint CPU components summed. No UTC estimates.',
                    nested_subtimers_added=False,new_operation_created_by_normalization=False)
        write(path,norm); p=proof(path);normalized.append(p)
        new_rows.append(dict(id=identity,category=entry.get('category','cpu_analysis'),source=p,
            status_pointer='/status',accepted_statuses=['normalized_actual_measured_operation'],wall_pointer='/wall_seconds',
            process_cpu_pointer='/recorded_inclusive_CPU_seconds',timing_semantics=semantics,
            add_to_recorded_wall_totals=is_parent,parent_id=None,canonical_id='explicit_timer:'+source['path']+'#'+entry['wall_pointer'],scope=entry['scope']))
        summary.append(dict(id=identity,category=entry.get('category','cpu_analysis'),source=source,wall_seconds=wall,
                            known_CPU_seconds=norm['recorded_inclusive_CPU_seconds'],additive_parent=is_parent))
    for entry in parents:
        normalize(entry,True)
    for entry in regions:
        normalize(entry,False)
    # Actual B completion/publication sources are binding evidence; their nested timers are not extra operations.
    B_completion=proof('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/B_all_final/completion.json')
    B_pub=proof('docs/B_all_final_weights_durable_publication.json')
    completed,published=read(B_completion),read(B_pub)
    assert completed['status']=='passed_complete_scratch_package_not_published' and completed['logical_endpoints']==completed['unique_weights']==60
    assert published['status']=='passed' and published['source_sha256']==published['destination_sha256']==completed['tar_sha256']
    extra_proofs=[B_completion,B_pub,proof('results/B_all_final_publication_metadata_acceptance_v1.json'),
                  proof('artifacts/provenance/C_native_final_restore_independent_review_v2/COMPLETE.json')]
    for source in candidates['metadata_proofs']+extra_proofs:
        read(source);original_metadata[str(Path(source['path']).resolve())]=source
    old_ids={v['id'] for v in spec['additional_operations']}
    assert not (old_ids & {v['id'] for v in new_rows})
    spec['additional_operations'] += new_rows
    existing={str(Path(v['path']).resolve()) for v in spec['metadata_sources']}
    spec['metadata_sources'] += [v for k,v in original_metadata.items() if k not in existing]
    spec['created_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat();spec['source_spec_revision']=7
    spec['preserved_v6_source_spec']=previous;spec['unchanged_v2_collector']=collector
    spec['revision7_candidate_source']=candidate_source
    spec['revision7_additional_parent_operation_ids']=[v['id'] for v in parents]
    spec['revision7_separate_region_operation_ids']=[v['id'] for v in regions]
    spec['revision7_normalized_sources']=normalized
    spec['revision7_source_preparation']=proof(__file__)
    spec['revision7_candidate_category_mapping_to_unchanged_collector_enum']=category_mapping
    spec['accounting_contract'].update(v1_through_v6_sources_and_collectors_unchanged=True,
        revision7_original_verify_output_hashes_policy_retained=True,
        revision7_score_output_streaming_SHA_bytes_reported_not_semantically_parsed=True,
        revision7_checkpoint_base_ZIP_TAR_payload_reads_forbidden=True,
        revision7_B_batch_sample_and_nested_assembly_not_double_charged=True,
        revision7_840_excludes_provenance_native29_timeout6=True,
        revision7_A_remaining_batch_Cbest_and_future_native_restore_not_imputed=True)
    spec['unmeasured_additional_gaps'] += [{'scope':v} for v in candidates['unmeasured_parent_gaps']]
    spec['unmeasured_additional_gaps'] += [dict(scope=v) for v in [
        'B full CPU analysis remains unknown; structured guard timing is only its measured child region.',
        'Native recovery preparation/independent review/timeout-repair full engineering parents unknown; only actual saved test/probe children counted.',
        'A remaining-final batch and publication not complete at registration freeze; excluded even if live progress later advances.',
        'No Cbest whole batch or original1B/2.8B native real restore has executed; no measured work imputed.',
        'Theory/thesis authoring, interactive defense slide writing/visual inspection and C_nll helper authoring whole parent timings unknown.',
    ]]
    spec['revision7_nested_source_exclusions']=[dict(source=B_completion,pointer='/wall_seconds',reason='assembly nested in B batch wholechild'),
        dict(source=B_pub,pointer='/wall_seconds',reason='internal publisher nested in B publisher wholechild'),
        dict(source=extra_proofs[-1],pointer='/independent_probe_complete_parent_timing',reason='same fixed probe referenced by COMPLETE, already counted once'),
        dict(source=proof('artifacts/provenance/C_native_final_restore_preparation_v1/timeout_repair_v1/repair_receipt.json'),
             pointer='/copied_end_to_end_toy_receipts',reason='inner toy receipts/copies nested in timeout6-test wholechild')]
    output=ROOT/'configs/cost_inventory_final_sources_v7.json';write(output,spec)
    report=dict(schema_version=1,status='v7_source_revision_actual_operations_only_not_whole_study_complete',
        spec=proof(output),preserved_v6=previous,collector=collector,parents=summary[:len(parents)],
        regions_separate_nonadditive=summary[len(parents):],
        parent_operations=len(parents),parent_wall_increment_seconds=sum(v['wall_seconds'] for v in summary if v['additive_parent']),
        parent_known_CPU_increment_seconds=sum(v['known_CPU_seconds'] or 0 for v in summary if v['additive_parent']),
        partial_region_wall_separate_seconds=sum(v['wall_seconds'] for v in summary if not v['additive_parent']),
        original_metadata_sources=list(original_metadata.values()),
        actual_B_complete_and_published=True,A_remaining_batch_and_Cbest_not_registered=True,
        dynamic_C_and_opaque_growth_are_separate_from_this_static_registration_increment=True)
    write(HERE/'registration_delta.json',report)
    print(json.dumps({k:report[k] for k in ['parent_operations','parent_wall_increment_seconds','parent_known_CPU_increment_seconds','partial_region_wall_separate_seconds']}))


if __name__=='__main__':
    main()
