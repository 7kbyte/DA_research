"""Correct the operation-cutoff semantics of the preserved 16-case preview."""
import csv
import hashlib
import json
import resource
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROV = Path(__file__).resolve().parent
OUT = ROOT / 'results/optional_weight_expansion_cutoff_corrigendum_v1'


def parse(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def main():
    started = time.perf_counter()
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    OUT.mkdir(exist_ok=False)
    inputs = PROV / 'inputs'
    inputs.mkdir(exist_ok=True)
    sources = {}
    def read(label, relative):
        p = ROOT / relative
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        sources[label] = {'path':str(p),'sha256':sha,'bytes':len(raw)}
        (inputs/(label+'_'+p.name)).write_bytes(raw)
        return raw
    raw = read('original_plan','results/optional_weight_expansion_planning_v1/plan.json')
    assert hashlib.sha256(raw).hexdigest() == '1fa0712a3ea4fc1dd6c61c8129f206966f7c4258dc505d28f971e2c8e957d698'
    old = json.loads(raw)
    read('original_producer','artifacts/provenance/optional_weight_expansion_planning_v1/produce.py')
    read('original_doc','docs/新增五包权重容量与CPU时间边界.md')
    exporter = read('current_exporter_read_only','scripts/export_additional_endpoints.py').decode()
    assert 'values[3] >= 120' in exporter
    assert 'until_publication_start = missing_unique*values[0] + values[1]' in exporter
    assert "now + dt.timedelta(seconds=until_publication_start) < DEADLINE" in exporter
    cutoff = parse(old['new_encoding_start_cutoff_utc_exclusive'])
    tail = parse(old['final_delivery_tail_starts_utc'])
    minimum_margin = 120
    cases = []; flat = []
    for index, case in enumerate(old['cases']):
        meta = {k:v for k,v in case.items() if k not in ['rows','all_five_time_gates_pass_under_assumptions']}
        rows = []
        for row in case['rows']:
            start = parse(row['first_encoding_start_utc_proxy'])
            encode_done = start + timedelta(seconds=row['roundtrip_wall_seconds_proxy'])
            assembly_done = encode_done + timedelta(seconds=row['assembly_wall_seconds_proxy'])
            finish = parse(row['package_durable_finish_utc_proxy'])
            publication_start = finish - timedelta(seconds=row['publication_wall_seconds_proxy'])
            assert abs((publication_start-assembly_done).total_seconds()-row['publication_serial_wait_seconds_proxy']) < .000004
            item = {'case_index':index,**meta,'id':row['id'],
                'original_last_encoding_start_utc_proxy':row['last_encoding_start_utc_proxy'],
                'actual_meaning_of_original_last_new_operation':'last encoder start only; excludes subsequent assembly operation',
                'all_encoding_complete_and_assembly_start_utc_proxy':encode_done.isoformat(),
                'assembly_complete_utc_proxy':assembly_done.isoformat(),
                'ROOT_publication_start_after_existing_simulated_lane_wait_utc_proxy':publication_start.isoformat(),
                'durable_finish_utc_proxy_unchanged':finish.isoformat(),
                'original_arithmetic_gates_pass':row['all_new_encodings_strictly_before0200'] and row['package_finishes_before30minute_delivery_tail'],
                'assembly_start_strictly_before0200':encode_done < cutoff,
                'whole_encoding_and_assembly_complete_strictly_before0200':assembly_done < cutoff,
                'publication_start_strictly_before0200_with_saved_lane_wait':publication_start < cutoff,
                'finish_plus_exporter_minimum_margin120s_before_tail':finish + timedelta(seconds=minimum_margin) <= tail,
                'additional_actual_ROOT_finish_margin_not_yet_selected':True}
            item['corrected_arithmetic_gates_pass_with_only_minimum_margin'] = all(item[k] for k in [
                'whole_encoding_and_assembly_complete_strictly_before0200',
                'publication_start_strictly_before0200_with_saved_lane_wait',
                'finish_plus_exporter_minimum_margin120s_before_tail'])
            flat.append(item); rows.append(item)
        cases.append({'case_index':index,**meta,
            'old_all_five_arithmetic_gates_pass':case['all_five_time_gates_pass_under_assumptions'],
            'corrected_all_five_arithmetic_gates_pass_with_only_minimum_margin':all(x['corrected_arithmetic_gates_pass_with_only_minimum_margin'] for x in rows),
            'rows':rows})
    assert len(cases) == 16 and len(flat) == 80
    bounds = []
    for old_bound in old['latest_C_ready_bounds_high1p25']:
        d = old['definitions'][old_bound['id']]
        n = d['conditional_D_alias_unique_count'] if old_bound['conditional_aliases'] else d['no_alias_unique_count']
        work = (n*d['parent_roundtrip_seconds_proxy']+d['assembly_seconds_proxy'])*1.25
        by_op = cutoff-timedelta(seconds=work)
        by_tail = tail-timedelta(seconds=work+d['publication_seconds_proxy']*1.25+minimum_margin)
        corrected = min(by_op, by_tail)
        # Recover the saved training complete time without recomputing its FIFO.
        old_bound_time = parse(old_bound['latest_first_start_exclusive_or_tail_bound_utc_proxy'])
        train_end = old_bound_time-timedelta(seconds=old_bound['extra_READY_delay_isolated_upper_headroom_seconds'])
        bounds.append({'variant':old_bound['variant'],'id':old_bound['id'],
            'conditional_aliases':old_bound['conditional_aliases'],
            'old_latest_first_start_utc_proxy':old_bound_time.isoformat(),
            'corrected_latest_first_start_exclusive_utc_proxy':corrected.isoformat(),
            'advance_seconds':(old_bound_time-corrected).total_seconds(),
            'saved_training_phase_complete_utc_proxy':train_end.isoformat(),
            'remaining_READY_headroom_seconds_proxy':(corrected-train_end).total_seconds(),
            'scope':'isolated high1.25 lower-margin boundary; extra READY latency, current IO contention, original Cfinal priority and real ROOT margin remain separate; not launch authority'})
    result={'schema_version':1,'status':'read_only_historical_planning_corrigendum_not_execution_authority',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'sources':sources,
        'original_16_cases_preserved':True,'case_count':16,'package_rows':80,
        'correction':'Original last-new-operation was last encoder start. It omitted the later assembly admission. The current conservative admission requires all remaining encoding plus assembly before02:00, and publication plus explicit margin by02:56:32.',
        'corrected_cutoff_utc_exclusive':cutoff.isoformat(),'protected_finish_tail_utc':tail.isoformat(),
        'exporter_minimum_finish_margin_seconds':minimum_margin,'minimum_margin_is_not_a_ROOT_approved_package_timing_plan':True,
        'durations_training_order_capacity_and_saved_publication_lane_unchanged':True,
        'no_new_training_preview_search_or_actual_READY_assumption':True,
        'cases':cases,'corrected_isolated_high1p25_bounds':bounds,
        'old_all_five_pass_count':sum(x['old_all_five_arithmetic_gates_pass'] for x in cases),
        'corrected_all_five_pass_count_with_only_minimum_margin':sum(x['corrected_all_five_arithmetic_gates_pass_with_only_minimum_margin'] for x in cases),
        'scripts_helper_queue_exporter_authority_or_GPU_changed':False,
        'score_or_large_payload_reads':False}
    (OUT/'corrigendum.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'all16cases_80packages.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    selected=[x for x in flat if x['id']=='C2p8B_best' and x['new_CPU_work_wall_factor']==1.25 and not x['alias_21_21_requires_actual_best_proofs']]
    lines=['# 新增五包权重：02:00操作门槛勘误','',
        '本勘误保留原16情景、原文和原producer字节。原表“最后一次新操作开始”实际是最后一份encoder开始；它没有包含随后新启动的assembly。因此原22:32:11孤立READY界过松，不应用作实际启动门槛。此处只纠正已有预测的时间语义，不生成新READY、authority或执行任务。','',
        '新保守门槛为：预计全部剩余encode＋assembly结束、即预计ROOT publication开始，必须严格早于02:00；其后publication和明确finish margin还须在02:56:32保护尾部前完成。当前exporter要求margin至少120s；本表仅使用这个已存在的下限作算术核对，并不代替实际资源验收和ROOT所选更大余量。保存情景中已有的publication排队也计入publication开始界。','',
        '| 已存训练情景/额外READY延迟 | 最后encoder开始（旧字段） | assembly完成/预计publication开始 | 整包发布完成（原预测不变） | 新算术门槛 |',
        '|---|---|---|---|---|']
    for row in selected:
        lines.append(f"| {row['training_order_variant']} / {row['additional_C_READY_delay_reserve_seconds']//60}min | {row['original_last_encoding_start_utc_proxy']} | {row['ROOT_publication_start_after_existing_simulated_lane_wait_utc_proxy']} | {row['durable_finish_utc_proxy_unchanged']} | {'通过（仅最小margin）' if row['corrected_arithmetic_gates_pass_with_only_minimum_margin'] else '不通过'} |")
    b=next(x for x in bounds if x['id']=='C2p8B_best' and not x['conditional_aliases'])
    lines += ['',f"在保留原×1.25代理、15份2.8B编码和assembly/publication速率的孤立估算中，C2.8B实际READY开工界由原 {b['old_latest_first_start_utc_proxy']} 提前到严格早于 {b['corrected_latest_first_start_exclusive_utc_proxy']}。已存C-only训练phase完成预测约22:07，已晚于该界；加早NLL的原假设情景更晚。因此不能继续引用原零READY延迟高情景的旧通过标记。",'',
        f"原16情景中五包全通过为{result['old_all_five_pass_count']}，修正后在仅120s最低margin下为{result['corrected_all_five_pass_count_with_only_minimum_margin']}。全部80包行和12个孤立C界均保存在JSON/CSV，可逐行追溯；未重新搜索顺序或更改任务数量、horizon、容量和任何预定科学工作。",'',
        '这些仍是旧时点、单样本full-final作为best代理的forecast，既不是best实测保证，也不是最新实际phase READY预测。Cbest尚未运行；后续必须按实际完整READY、已验证资源样例和当时完整剩余包成本重新核算，并保全原任务及30min交付尾部。已启动的一次operation可自然结束，并不授权在02:00以后新启动assembly。','']
    (ROOT/'docs/新增五包权重02时门槛勘误.md').write_text('\n'.join(lines))
    assert hashlib.sha256((ROOT/'results/optional_weight_expansion_planning_v1/plan.json').read_bytes()).hexdigest()==sources['original_plan']['sha256']
    after=resource.getrusage(resource.RUSAGE_SELF)
    receipt={'schema_version':1,'status':'passed','producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wall_seconds':time.perf_counter()-started,'user_cpu_seconds':after.ru_utime-cpu.ru_utime,'sys_cpu_seconds':after.ru_stime-cpu.ru_stime,
        'timer_scope':'post-import correction body and writes; not whole parent or actual export cost',
        'checks':{'all16cases_80rows_retained':True,'original_plan_sha_unchanged':True,'arithmetic_input_only_no_live_state_or_GPU':True},
        'outputs':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [OUT/'corrigendum.json',OUT/'all16cases_80packages.csv',ROOT/'docs/新增五包权重02时门槛勘误.md']]}
    (PROV/'completion_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'old_pass':result['old_all_five_pass_count'],'corrected_pass':result['corrected_all_five_pass_count_with_only_minimum_margin'],'C2p8_new_latest':b['corrected_latest_first_start_exclusive_utc_proxy'],'outputs':receipt['outputs']},ensure_ascii=False))


if __name__ == '__main__':
    main()
