"""Explicit append-only v8 metadata source registration; unchanged collectors."""
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
V7_SHA = '0cab816da531e857ded88a004624d9a946a1bd3dae75f295ed68e1e4f40b1f0e'
COLLECTOR_SHA = '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae'
AUDIT_SHA = 'a3513d44a64adbab37feafcb1f7d0e15a31a61defac7572cd5482621775bede9'
CATEGORIES = {'cpu_analysis','opaque_prehash','lossless_export','storage_publication','archive_qa','storage_wait_and_reservation'}
ORIGINALS = {}


def proof(path):
    path = (ROOT / path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(source):
    path = Path(source['path'])
    assert path.suffix == '.json' and path.stat().st_size <= 32 << 20
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source['sha256'], str(path)
    return json.loads(raw)


def ptr(value, path):
    for key in path.split('/')[1:]:
        key = key.replace('~1','/').replace('~0','~')
        value = value[int(key)] if isinstance(value,list) else value[key]
    return value


def new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def category(identity):
    if identity in {'A_remaining93_batch','C410_best_first_sample','C410_best_batch','snapshot1918_build'}:
        return 'lossless_export'
    if identity.startswith('B_native_') or identity.startswith('Cbest_') or 'bounded' in identity or 'report_QA' in identity:
        return 'archive_qa'
    if 'publication' in identity or identity == 'A_remaining93_metadata_review':
        return 'storage_publication'
    if identity == 'snapshot1918_allocation_region':
        return 'storage_wait_and_reservation'
    return 'cpu_analysis'


def main():
    start = time.monotonic()
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    previous = proof('configs/cost_inventory_final_sources_v7.json'); assert previous['sha256'] == V7_SHA
    collector = proof('scripts/inventory_study_costs_final_v2.py'); assert collector['sha256'] == COLLECTOR_SHA
    audit = proof('artifacts/provenance/cost_sources_v7/run_collection_with_IO_audit.py'); assert audit['sha256'] == AUDIT_SHA
    spec = read(previous)
    old_operations = copy.deepcopy(spec['additional_operations'])
    registered = {}
    registered_sha = {}
    for row in old_operations:
        s = row['source']; obj = read(s)
        raw = obj.get('original_source',s)
        wall = obj.get('original_wall_pointer',row.get('wall_pointer'))
        registered[(str(Path(raw['path']).resolve()),wall)] = row['id']
        registered_sha[(raw['sha256'],wall)] = row['id']
    pending = 'artifacts/provenance/cost_sources_v8_pending_v1/'
    catalogs = [proof(pending + name) for name in ['pending_sources.json','supplement_01_completed_C410_best.json','supplement_02_C410_best_publication_region.json']]
    expected_catalogs = ['aadb3e80993940f32aef2c1d0f0cf744185d7b4376c78bb3fd8bd112264d0da5','9b4f44cdbba805edc5a864775cb574164cb89932bfc20408752ad882c68aa44c','69bcbf449c03f834c1b5c7a513a7bffcbc0400b73570598625cd50516058eeef']
    descriptors = []
    for source, expected in zip(catalogs,expected_catalogs):
        assert source['sha256'] == expected
        ORIGINALS[source['path']] = source
        descriptors.extend(read(source)['candidates'])
    assert len(descriptors) == 39
    def add(identity,path,wall,cpus,scope,*,region=False,rc='/returncode'):
        source = proof(path)
        obj = read(source)
        descriptors.append({'id':identity,'source':source,'wall_pointer':wall,
            'cpu_component_pointers':cpus,'scope':scope,
            'classification':'pending_measured_region_parent_unknown' if region else 'pending_measured_independent_parent',
            'returncode_pointer':rc,'returncode':ptr(obj,rc) if rc else None})
    for n in (1,2,3):
        add(f'report_delivery_section7_render_v{n}',f'artifacts/provenance/report_delivery_section7_update_v{n}/render_whole_child_timing.json',
            '/whole_child_wall_seconds',['/whole_child_user_cpu_seconds','/whole_child_sys_cpu_seconds'],
            'Independent actual report render spawn-to-exit. V2 was a redundant no-source-change render; original inaccurate scope corrected only by retained correction_receipt.json. All actual costs retained, internal render timers nested.')
    for n in (1,3):
        add(f'report_QA_delivery_section7_v{n}',f'artifacts/report_review/delivery_section7_update_v{n}/whole_child_timing.json',
            '/whole_child_wall_seconds',['/whole_child_user_cpu_seconds','/whole_child_sys_cpu_seconds'],
            'Independent actual PDF/HTML/raster QA child; manual visual inspection whole unknown. Not nested in render.')
    add('snapshot1918_build','artifacts/provenance/snapshot_1918_build/whole_process_timing.json','/whole_child_wall_seconds',
        ['/whole_child_user_cpu_seconds','/whole_child_sys_cpu_seconds'],'Actual whole package build child including compression and all-payload verification; inner packager timers nested.')
    add('snapshot1918_publication','artifacts/provenance/snapshot_1918_publication/whole_child_timing.json','/wall_seconds',
        ['/user_cpu_seconds','/sys_cpu_seconds'],'Actual whole publisher child. Internal source/copy/readback/fsync timer nested.')
    add('snapshot1918_allocation_region','artifacts/provenance/snapshot_1918_publication/actual_allocation.json','/wall_seconds',
        ['/user_cpu_seconds','/sys_cpu_seconds'],'Actual allocation/chown/fsync region only; human publication parent unknown.',region=True,rc=None)
    for n in (1,2):
        add(f'snapshot1918_bounded_review_attempt{n}',f'artifacts/provenance/snapshot_1918_independent_bounded_review_v{n}/whole_child_timing.json',
            '/whole_child_wall_seconds',['/child_user_seconds','/child_system_seconds'],
            'Separate actual bounded project review child. Attempt1 failed at PDF API on a partial gzip traversal; compressed bytes unknown. Attempt2 passed full forward stream784476267 bytes, selected32 plus manifest SHA; no weight archive/CP/base. Internal main timer nested.')
    for n in (1,2):
        add(f'evaluation_budget_v6_refresh1925_attempt{n}',f'artifacts/provenance/evaluation_budget_v6_refresh_1925_v{n}/whole_child_timing.json',
            '/whole_child_wall_seconds',['/user_cpu_seconds','/sys_cpu_seconds'],
            'Independent actual score-blind budget producer child; attempt1 rc3 failure retained, attempt2 corrected success retained.')
    add('evaluation_budget1925_independent_description','artifacts/provenance/evaluation_budget_v6_refresh_1925_v2/whole_description_timing.json',
        '/whole_child_wall_seconds',['/user_cpu_seconds','/sys_cpu_seconds'],'Separate actual score-blind description/check child; not nested in producer.')
    add('C410_best_publication_independent_actual_check','artifacts/provenance/C410_best_publication_independent_acceptance_v1/whole_independent_check_timing_attempt1.json',
        '/wall_seconds',['/user_seconds','/system_seconds'],'Independent actual small metadata/stat publication checker child. Manual review/authoring whole unknown; not another publisher.')
    for leaf,identity in [('preparation_whole_child_timing.json','pending_v8_sources_attempt1_failed'),('preparation_attempt2_whole_child_timing.json','pending_v8_sources_attempt2_success')]:
        add(identity,pending+leaf,'/whole_child_wall_seconds',['/child_user_seconds','/child_system_seconds'],
            'Actual pending-source preparation child only; failed first attempt and corrected successful attempt separate. Source/log bytes retained. Inner preparation region nested; preceding manual reading/authoring unknown.')
    add('pending_v8_sources_finalization_region',pending+'finalization_region_receipt.json','/region_wall_seconds',
        ['/region_self_user_seconds','/region_self_system_seconds'],'Independent small metadata finalization region; no complete manual parent timer.',region=True,rc=None)
    add('delivery_guide_v6_union_region','results/delivery_capacity_guide_v6/metadata_calculation.json','/timing/wall_seconds',
        ['/timing/self_user_seconds','/timing/self_sys_seconds'],'Actual small eight-manifest/stat/normalization region; authoring whole unknown. Copy in guide source excluded.',region=True,rc=None)
    add('delivery_guide_v6_navigation_region','docs/交付文件与恢复容量指南_来源核对_v6.json','/measured_regions/guide_and_navigation_region/wall_seconds',
        ['/measured_regions/guide_and_navigation_region/self_user_seconds','/measured_regions/guide_and_navigation_region/self_sys_seconds'],
        'Guide/navigation writing small region only; full engineering parent unknown.',region=True,rc=None)
    add('delivery_guide_v6_source_QA_region','results/delivery_capacity_guide_v6/source_QA.json','/timer/region_wall_seconds',
        ['/timer/region_self_user_seconds','/timer/region_self_system_seconds'],'Separate small-source SHA QA region, not the whole guide task.',region=True,rc=None)
    add('delivery_guide_v7_snapshot_update_region','docs/交付文件与恢复容量指南_来源核对_v7.json','/measured_region/wall_seconds',
        ['/measured_region/self_user_seconds','/measured_region/self_sys_seconds'],'Separate small metadata/stat/navigation update region; bounded archive review is independently charged, no overlap.',region=True,rc=None)

    rows, parents, regions, seen, skipped = [], [], [], set(), []
    for entry in descriptors:
        raw_source = proof(entry['source']['path'])
        assert raw_source['sha256'] == entry['source']['sha256']
        obj = read(raw_source)
        wallptr = entry['wall_pointer']; pathkey = (raw_source['path'],wallptr); shakey = (raw_source['sha256'],wallptr)
        if pathkey in registered or shakey in registered_sha:
            skipped.append({'candidate_id':entry['id'],'source':raw_source,'pointer':wallptr,'existing_id':registered.get(pathkey,registered_sha.get(shakey))})
            continue
        assert pathkey not in seen and shakey not in seen, 'duplicate explicit timer source'
        seen.add(pathkey);seen.add(shakey)
        wall = ptr(obj,wallptr); cpus = entry.get('cpu_component_pointers',[])
        components = {p:ptr(obj,p) for p in cpus}
        assert all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in [wall,*components.values()])
        rcp = entry.get('returncode_pointer'); rc = ptr(obj,rcp) if rcp else None
        if rcp:
            assert type(rc) is int and rc == entry['returncode']
        is_parent = entry['classification'] == 'pending_measured_independent_parent'
        semantics = 'independent_measured_complete_operation' if is_parent else 'partial_region_parent_unknown_not_additive'
        identity = 'v8_' + entry['id']; cat = category(entry['id']); assert cat in CATEGORIES
        value = {'schema_version':1,'status':'normalized_actual_measured_operation','original_source':raw_source,
            'original_wall_pointer':wallptr,'wall_seconds':wall,'CPU_component_pointers':cpus,'CPU_components':components,
            'recorded_inclusive_CPU_seconds':sum(components.values()) if cpus else None,
            'original_returncode':rc,'scope':entry['scope'],'timing_semantics':semantics,
            'normalization':'Measured wall copied; only named disjoint CPU components summed. Unknowns remain null; failed actual invocations preserve rc. No UTC estimates.',
            'nested_subtimers_added':False,'new_operation_created_by_normalization':False}
        path = HERE / (identity+'_normalized.json');new(path,value);norm_source = proof(path)
        rows.append({'id':identity,'category':cat,'source':norm_source,'status_pointer':'/status',
            'accepted_statuses':['normalized_actual_measured_operation'],'wall_pointer':'/wall_seconds',
            'process_cpu_pointer':'/recorded_inclusive_CPU_seconds','timing_semantics':semantics,
            'add_to_recorded_wall_totals':is_parent,'parent_id':None,
            'canonical_id':'explicit_timer:'+raw_source['path']+'#'+wallptr,'scope':entry['scope']})
        ORIGINALS[raw_source['path']] = raw_source
        (parents if is_parent else regions).append({'id':identity,'category':cat,'source':raw_source,'wall_pointer':wallptr,
            'wall_seconds':wall,'known_CPU_seconds':value['recorded_inclusive_CPU_seconds'],'original_returncode':rc,'additive_parent':is_parent})
    assert not skipped, 'Unexpected old-v7 duplicate; require explicit review rather than silently omit'
    extra = ['artifacts/provenance/report_delivery_section7_update_v2/correction_receipt.json',
             'artifacts/provenance/snapshot_1918_independent_bounded_review_v2/COMPLETE.json',
             'artifacts/provenance/C410_best_publication_actual_v1/outer_timing_receipt_failure.json',
             'artifacts/provenance/C410_best_publication_actual_v1/ROOT_acceptance.json',
             'artifacts/provenance/C410_best_publication_independent_acceptance_v1/COMPLETE.json',
             'results/delivery_capacity_guide_v6/initial_failed_calculation.json']
    for path in extra:
        source = proof(path);read(source);ORIGINALS[source['path']] = source
    spec['additional_operations'] += rows
    assert spec['additional_operations'][:len(old_operations)] == old_operations
    existing = {str(Path(x['path']).resolve()) for x in spec['metadata_sources']}
    spec['metadata_sources'] += [s for path,s in ORIGINALS.items() if path not in existing]
    spec.update(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_spec_revision=8,
        preserved_v7_source_spec=previous,unchanged_v2_collector=collector,
        revision8_unchanged_IO_audit_wrapper=audit,revision8_pending_catalogs=catalogs,
        revision8_source_preparation=proof(__file__),revision8_additional_parent_operation_ids=[x['id'] for x in parents],
        revision8_separate_region_operation_ids=[x['id'] for x in regions],revision8_normalized_sources=[x['source'] for x in rows])
    spec['accounting_contract'].update(revision8_v1_through_v7_and_collectors_unchanged=True,
        revision8_legacy_enum_and_verify_output_hashes_true_unchanged=True,
        revision8_C410best_publisher_external_wall_CPU_childrc_unknown=True,
        revision8_copied1953_batch_timer_not_publication=True,
        revision8_B_native_single_outer36second_parent_inner_nested=True,
        revision8_this_successful_collection_cost_retained_externally_not_self_charged=True)
    spec['revision8_historical_gap_status_resolutions'] = [
        'Historical v7-freeze statements that A93 batch/C410best batch had not completed are retained as historical source bytes; both now have explicit actual completed v8 parents.',
        'C410best publication completed durably; exact publisher outer whole/CPU/childrc unavailable after wrapper receipt collision. Only publisher internal195.216953 seconds is a nonadditive region.',
        'B fixed native restore completed exactly one prespecified draw; its36.184441s external parent includes all inner wrapper/body/extraction/decode/fingerprint timers.',
        '1918 reviewer failed partial first traversal retains actual parent cost and unknown compressed-byte scope; successful full stream is a separate actual call.',
        'V6 guide failed initial metadata calculation has no saved complete parent timer; manual whole stays unknown.',
        'Current v8 collection/normalization/preparation engineering whole not charged to itself. Dynamic C final records are independently counted only if completed and source-bound at observed snapshot; no future work imputed.'
    ]
    config = ROOT/'configs/cost_inventory_final_sources_v8.json';new(config,spec)
    legacy = read(read(spec['v1_source_spec'])['legacy_source_spec'])
    assert all(x['verify_output_hashes'] is True for x in legacy['evaluation_manifests'])
    delta = {'schema_version':1,'status':'v8_explicit_measured_source_registration_not_whole_study_final',
        'spec':proof(config),'preserved_v7':previous,'collector':collector,'IO_audit_wrapper':audit,
        'parents':parents,'regions_separate_nonadditive':regions,'parent_operations':len(parents),'region_operations':len(regions),
        'parent_wall_increment_seconds':sum(x['wall_seconds'] for x in parents),
        'parent_known_CPU_increment_seconds':sum(x['known_CPU_seconds'] or 0 for x in parents),
        'partial_region_wall_separate_seconds':sum(x['wall_seconds'] for x in regions),
        'source_path_and_SHA_pointer_dedup_checked_against_all_v7_operations':True,'skipped_existing_duplicates':skipped,
        'original_metadata_sources':list(ORIGINALS.values()),'dynamic_C_opaque_and_training_growth_separate_from_static_registration_increment':True,
        'whole_study_complete':False,'current_collection_self_cost_registered':False,
        'preparation_region':{'wall_seconds':time.monotonic()-start,'self_user_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime-cpu.ru_utime,'self_system_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_stime-cpu.ru_stime,
            'scope':'Only this preparation script main-entry through normalization/config/source reads; manual authoring and final delta serialization excluded. Not included in v8 ledger.'}}
    new(HERE/'registration_delta.json',delta)
    print(json.dumps({k:delta[k] for k in ['parent_operations','region_operations','parent_wall_increment_seconds','parent_known_CPU_increment_seconds','partial_region_wall_separate_seconds']}))


if __name__ == '__main__':
    main()
