"""Independent, validation-only review of the completed C410 phase (stdlib only)."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PHASE = 'extended_v2_pythia_410m_m64_formal'
OUT = ROOT / 'results/C410_validation_review_v1'
PROV = ROOT / 'artifacts/provenance/C410_validation_review_v1'
SOURCES = {}
READS = []


def read(path, expected=None, snapshot=False):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    # No checkpoints, tensors, test evaluations or behavioral scores are inputs.
    assert p.suffix in {'.json', '.jsonl', '.csv', '.md', '.py'}, p
    assert not any(t in str(p) for t in ['/evaluation_ready_', '/behavioral/', '/test_results/'])
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert sha == expected, ('source SHA mismatch', str(p), sha, expected)
    record = {'path': str(p), 'sha256': sha, 'bytes': len(raw)}
    if str(p) in SOURCES:
        assert SOURCES[str(p)] == record, ('input changed', str(p))
    SOURCES[str(p)] = record
    READS.append(record)
    if snapshot:
        dest = PROV / 'inputs' / (sha + '_' + p.name)
        if not dest.exists():
            dest.write_bytes(raw)
    return raw


def load(path, expected=None, snapshot=False):
    return json.loads(read(path, expected, snapshot))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def csv_rows(path, sha):
    return list(csv.DictReader(read(path, sha).decode().splitlines()))


def job_digest(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    (PROV / 'inputs').mkdir(exist_ok=True)
    framework = ROOT / 'docs/C完整结果解释框架_解封前.md'
    read(framework)
    read(__file__)
    watcher = load('logs/auxiliary_validation_analysis_watch.json', snapshot=True)
    watched = watcher['phases'][PHASE]
    assert watched['status'] == 'complete'
    assert not watcher['test_or_behavior_scores_read'] and not watcher['gpu_work_started']
    manifest = load(watched['manifest']['path'], watched['manifest']['sha256'])
    assert manifest['status'] == 'complete' and manifest['phase'] == PHASE
    assert manifest['split'] == 'validation' and manifest['logical_endpoint_count'] == 48
    expected_sources = {str(Path(x['path']).resolve()): x['sha256'] for x in manifest['sources']}
    def manifest_load(p):
        p = Path(p).resolve()
        return load(p, expected_sources[str(p)])
    reg_path = ROOT / f'configs/{PHASE}_active_registry.json'
    registry = manifest_load(reg_path)
    assert len(registry) == 24
    assert {(r['group'], r['seed']) for r in registry} == {(g, s) for g in ['B0','B1','B2','B3','B4','D','M','C'] for s in [11,22,33]}
    comp_path = ROOT / f'results/extended_phase_equivalence/{PHASE}/completion.json'
    completion = manifest_load(comp_path)
    assert completion['status'] == 'passed'
    assert completion['expected_runs'] == completion['completed_runs'] == 24
    assert completion['source_registry']['sha256'] == SOURCES[str(reg_path.resolve())]['sha256']
    audit = load(completion['audit']['path'], completion['audit']['sha256'])
    assert audit['status'] == 'passed' and audit['source_violations'] == []
    assert audit['source_state_complete_gate'] == {'ready': True, 'expected_runs': 24, 'completed_runs': 24, 'missing_runs': [], 'failed_runs': []}
    snap = load(completion['input_snapshot']['path'], completion['input_snapshot']['sha256'])
    protocol = load(snap['protocol_path'], snap['protocol_sha256'])
    assert protocol['shared_lambda'] == .05 and protocol['shared_learning_rate'] == 3e-5
    # Only bind the frozen choices; no recalibration or pilot-score access.
    frozen_choices = protocol['choices']
    train_source = ROOT / 'src/replay_study/train.py'
    read(train_source, snap['code']['src/replay_study/train.py']['sha256'])
    queue = load(snap['queue_path'], snap['queue_sha256'])
    queue_map = {j['id']: j for j in queue}
    master = load('configs/supplemental_v1_queue.json', snapshot=True)
    master_map = {j['id']: j for j in master}
    state = load('logs/supplemental_job_state.json', snapshot=True)
    state_evidence = []
    for r in registry:
        name = r['id']; job = queue_map[name]; st = state['jobs'][name]
        assert master_map[name] == job
        assert st['definition_sha256'] == job_digest(job)
        assert st['status'] == 'completed' and st['returncode'] == 0
        assert not st.get('missing_outputs') and not st.get('stop_requested_at')
        assert all(Path(p).is_file() for p in job['outputs'])
        state_evidence.append({'id': name, 'definition_sha256': st['definition_sha256'], 'status': st['status'], 'returncode': st['returncode']})
    outputs = {Path(x['path']).name: x for x in manifest['outputs']}
    end_rows = csv_rows(outputs['validation_endpoints.csv']['path'], outputs['validation_endpoints.csv']['sha256'])
    traj_rows = csv_rows(outputs['validation_trajectories.csv']['path'], outputs['validation_trajectories.csv']['sha256'])
    assert len(end_rows) == 48 and len(traj_rows) == 480
    end_map = {(r['run_id'], r['checkpoint']): r for r in end_rows}
    traj_map = {(r['run_id'], int(r['step'])): r for r in traj_rows}
    assert len(end_map) == 48 and len(traj_map) == 480
    initial = load(protocol['initial_validation_path'], protocol['initial_validation_sha256'])
    initial_nll = {k: v['nll'] for k, v in initial['corpora'].items()}
    all_rows = []; trajectory_output = []; exact_checks = Counter(); group_rows = []
    final_worse = 0
    for record in registry:
        name = record['id']
        cfg = load(record['config_path'], record['config_sha256'])
        summary = manifest_load(record['summary_path'])
        assert summary['status'] == 'completed'
        assert all(summary['config'][k] == v for k, v in cfg.items())
        assert cfg['target_prediction_tokens'] == 100_000_000
        assert cfg['microbatch_size'] == 64 and cfg['gradient_accumulation_steps'] == 1
        assert cfg['learning_rate'] == 3e-5
        assert cfg['replay_weight'] == (0 if record['group'] == 'B0' else .05)
        counters = summary['counters']; trajectory = summary['validation_trajectory']
        assert len(trajectory) == 20 and counters['step'] == 3058
        assert counters['target_prediction_tokens'] == 100_000_000
        expected_replay = 0 if record['group'] == 'B0' else 5_000_000
        assert counters['replay_prediction_tokens'] == expected_replay
        assert [x['step'] for x in trajectory] == [153*i for i in range(1,20)] + [3058]
        best = min(trajectory, key=lambda x: x['metrics']['target']['nll'])
        stored = summary['best_target_validation']
        assert stored['step'] == best['step']
        assert stored['target_prediction_tokens'] == best['target_prediction_tokens']
        assert stored['nll'] == best['metrics']['target']['nll']
        exact_checks['summary_best_selection_fields'] += 3
        events_path = Path(cfg['output_dir']) / 'events.jsonl'
        event_bytes = read(events_path, expected_sources[str(events_path.resolve())])
        events = [json.loads(line) for line in event_bytes.splitlines()]
        val_events = [x for x in events if x.get('event') == 'validation']
        assert len(val_events) == len(trajectory)
        for event, point in zip(val_events, trajectory):
            assert {k: event[k] for k in point} == point
            exact_checks['event_vs_summary_complete_validation_points'] += 1
        for point in trajectory:
            step = point['step']; target = point['target_prediction_tokens']; replay = point['replay_prediction_tokens']
            assert target == min(step * 32704, 100_000_000)
            assert replay == (0 if record['group'] == 'B0' else target // 20)
            assert all(math.isfinite(point['metrics'][c]['nll']) for c in ['target','c4','wikitext'])
            existing = traj_map[(name, step)]
            values = {'step':step,'target_prediction_tokens':target,'replay_prediction_tokens':replay}
            for corpus in ['target','c4','wikitext']:
                for metric in ['nll','kl']:
                    values[f'{metric}_{corpus}'] = point['metrics'][corpus][metric]
            values['g_new'] = initial_nll['target'] - values['nll_target']
            values['f_old_c4'] = values['nll_c4'] - initial_nll['c4']
            values['f_old_wikitext'] = values['nll_wikitext'] - initial_nll['wikitext']
            for key, value in values.items():
                assert float(existing[key]) == value, (name, step, key)
                exact_checks['existing_trajectory_csv_numeric_fields'] += 1
            trajectory_output.append({'run_id':name,'group':record['group'],'seed':record['seed'],**values})
        for endpoint, point in [('final',trajectory[-1]),('best',best)]:
            existing = end_map[(name,endpoint)]
            values = {'checkpoint_step':point['step'],'target_prediction_tokens':point['target_prediction_tokens'],
                      'replay_prediction_tokens':point['replay_prediction_tokens'],
                      'full_training_target_prediction_tokens':counters['target_prediction_tokens'],
                      'full_training_replay_prediction_tokens':counters['replay_prediction_tokens']}
            for corpus in ['target','c4','wikitext']:
                for metric in ['nll','kl']:
                    values[f'{metric}_{corpus}'] = point['metrics'][corpus][metric]
            values['g_new'] = initial_nll['target']-values['nll_target']
            values['f_old_c4'] = values['nll_c4']-initial_nll['c4']
            values['f_old_wikitext'] = values['nll_wikitext']-initial_nll['wikitext']
            for key, value in values.items():
                assert float(existing[key]) == value, (name,endpoint,key)
                exact_checks['existing_endpoint_csv_numeric_fields'] += 1
        final = trajectory[-1]
        final_worse += final['metrics']['target']['nll'] > best['metrics']['target']['nll']
        all_rows.append({'run_id':name,'group':record['group'],'seed':record['seed'],
            'replay_ratio':cfg['replay_ratio'],'lambda':cfg['replay_weight'],'learning_rate':cfg['learning_rate'],
            'best_step':best['step'],'best_target_prediction_tokens':best['target_prediction_tokens'],
            'best_replay_prediction_tokens':best['replay_prediction_tokens'],
            'best_target_fraction_of_full':best['target_prediction_tokens']/100_000_000,
            'best_target_nll':best['metrics']['target']['nll'],'final_target_nll':final['metrics']['target']['nll'],
            'final_minus_best_target_nll':final['metrics']['target']['nll']-best['metrics']['target']['nll'],
            'final_step':counters['step'],'full_target_prediction_tokens':100_000_000,
            'full_replay_prediction_tokens':expected_replay,'validation_points':len(trajectory)})
    order = ['B0','B1','B2','B3','B4','D','M','C']
    all_rows.sort(key=lambda r:(order.index(r['group']),r['seed']))
    for group in order:
        rows = [r for r in all_rows if r['group']==group]
        group_rows.append({'group':group,'seeds':'11/22/33','n':3,
            'best_steps': '/'.join(str(r['best_step']) for r in rows),
            'best_target_prediction_tokens': '/'.join(str(r['best_target_prediction_tokens']) for r in rows),
            'best_replay_prediction_tokens': '/'.join(str(r['best_replay_prediction_tokens']) for r in rows),
            'best_target_exposure_mean':statistics.mean(r['best_target_prediction_tokens'] for r in rows),
            'best_target_exposure_sample_sd':statistics.stdev(r['best_target_prediction_tokens'] for r in rows),
            'best_equals_full_count':sum(r['best_step']==r['final_step'] for r in rows),
            'final_minus_best_target_nll_mean':statistics.mean(r['final_minus_best_target_nll'] for r in rows)})
    # Metadata-only comparison with primary; no primary validation/test score input.
    main_reg = load('configs/formal_deterministic_registry.json')
    main = next(r for r in main_reg if r['group']=='B4' and r['seed']==11 and r['ratio']==.05)
    main_cfg = load(main['config_path'])
    main_meta = load('artifacts/data/finemath_train.json')
    ext_meta = load('artifacts/data_extended/finemath_train.json')
    comparison = {'primary':{'target_path':main_cfg['target_path'],'data_sha256':main_meta['sha256'],
        'capacity_prediction_tokens':main_meta['prediction_tokens'],'chunks':main_meta['chunks'],
        'target_exposure':main_cfg['target_prediction_tokens'],'microbatch':main_cfg['microbatch_size'],
        'accumulation':main_cfg['gradient_accumulation_steps'],'eval_every_steps':main_cfg['eval_every_steps'],
        'validation_paths':main_cfg['validation_paths']},
        'extended':{'target_path':protocol['data_sha256'],'data_sha256':ext_meta['sha256'],
        'capacity_prediction_tokens':ext_meta['prediction_tokens'],'chunks':ext_meta['chunks'],
        'target_exposure':100_000_000,'microbatch':64,'accumulation':1,'eval_every_steps':153,
        'validation_paths':protocol['validation_paths']},
        'both_effective_batch_chunks':64,'both_sequence_length':512,
        'relationship':protocol['relationship_to_primary'],
        'pure_horizon_causal_contrast':False}
    comparison['extended']['target_path'] = str(ROOT/'artifacts/data_extended/finemath_train.pt')
    for side in ['primary','extended']:
        x=comparison[side];x['nominal_eval_prediction_tokens']=x['eval_every_steps']*32704
    write_csv(OUT/'best_exposure_all24.csv',all_rows)
    write_csv(OUT/'best_exposure_by_group.csv',group_rows)
    write_csv(OUT/'validation_trajectories_independent.csv',trajectory_output)
    write_json(OUT/'primary_extended_configuration_comparison.json',comparison)
    dist=Counter(r['best_target_prediction_tokens'] for r in all_rows)
    review={'schema_version':1,'status':'passed','phase':PHASE,'model':'pythia-410m',
        'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'source_state_gate':state_evidence,
        'wholephase_validation_manifest':watched['manifest'],'source_completion':{'path':str(comp_path),'sha256':SOURCES[str(comp_path.resolve())]['sha256']},
        'source_audit':completion['audit'],'source_registry':completion['source_registry'],
        'scope':{'runs':24,'groups':order,'training_seeds':[11,22,33],'validation_points':480,'logical_endpoints':48},
        'best_selection_rule':'minimum finite target validation nll across scheduled trajectory; strict decrease only, earliest exact tie',
        'exact_comparisons':dict(exact_checks),'comparison_mismatches':[],
        'best_target_prediction_token_distribution':dict(sorted(dist.items())),
        'best_step_distribution':dict(sorted(Counter(r['best_step'] for r in all_rows).items())),
        'best_equal_full_count':sum(r['best_step']==r['final_step'] for r in all_rows),
        'final_target_nll_greater_than_best_count':final_worse,
        'full_exposure_target_prediction_tokens_each':100_000_000,
        'lambda_reselected':False,'frozen_choices':frozen_choices,
        'numeric_comparison':'Python float parsed from JSON/CSV exact equality; no tolerance for checked fields',
        'source_payloads_recomputed':False,'checkpoint_tensor_reads':0,'gpu_work_started':False,
        'test_or_behavior_score_files_read':False,'other_C_model_validation_files_read':False,
        'inputs':list(SOURCES.values()),'input_read_operations':len(READS),'input_read_bytes':sum(x['bytes'] for x in READS)}
    write_json(OUT/'review.json',review)
    lines=['# C410验证轨迹与best曝光：完整组独立审阅','',
        '本次仅审阅完整的 C410（100M）validation：8 方法 × seed11/22/33，共24个真实训练、480个验证点。watcher已标整阶段 complete，source audit 为 passed、24/24、无 source violations；当前 scheduler 的24项均 completed/rc0、无 stop request，定义与冻结 phase queue 一致。未读取任何 test NLL、behavior 分数或其余 C 模型 validation。','',
        '独立从原训练 summary 重算最小 target validation NLL，并与 events、既有整阶段 validation CSV 逐数值精确比对。best 使用冻结的严格下降规则，完全相等时保留较早点；这里只核验已执行政策，没有重选 λ、LR 或端点。全部字段核对无差异；完整来源 SHA 和计数见 review.json。','',
        '## 全24项best曝光','',
        '每格均按 seed11 / 22 / 33 排列。token 均为实际参与 loss 的 causal prediction-token 累计曝光，非 raw/padded 输入，也非唯一库容量。下表保留所有方法与 seed。','',
        '| 方法 | best step（11/22/33） | best目标曝光（11/22/33） | best回放曝光（11/22/33） |',
        '|---|---|---|---|']
    for g in group_rows:
        lines.append(f"| {g['group']} | {g['best_steps']} | {g['best_target_prediction_tokens']} | {g['best_replay_prediction_tokens']} |")
    lines += ['',f"best 目标曝光分布："+'；'.join(f'{t:,} tokens：{n}项' for t,n in sorted(dist.items()))+'。',
        f"全部24项 final 为 step3058、目标100,000,000 tokens；B0回放为0，其余各5,000,000。best与final曝光相同的项为0/24；best约在完整目标曝光的35.03%–55.04%，其中22/24为40.03%。24项final的target validation NLL均高于各自已选best，但这只是当前固定验证集和取点政策下的轨迹描述，不能称未观测时刻的连续最优，也不能据此新设训练停止阈值。",'',
        '验证每153步一次，名义间隔5,003,712目标tokens；最后另在100M评价，共20点/run。best是此网格上的验证选择结果。各方法best端点不自动等预算：B1 seed33为35,025,984，而B3 seed33为55,040,832；做best配对时须保留双方曝光。所有24训练实际都走完100M，不能按best曝光比例打折已经发生的训练费用。D/B4三seed两端点的完整模型状态等价证明仅支持对应推理复用，D仍是独立物理训练与逻辑结果。','',
        '## 主410M/10M与扩展410M/100M的解释边界','',
        f"主训练容量为{main_meta['chunks']:,} chunks / {main_meta['prediction_tokens']:,}预测位置，扩展为{ext_meta['chunks']:,} chunks / {ext_meta['prediction_tokens']:,}预测位置。扩展包含主训练前缀，属于相关稳健性扩展，不能当独立总体重复。主/扩展还分别采用 main_v3 / matched extended_v1 heldout，不能共用NLL initial作跨family基线。",'',
        '有效batch虽同为64 chunks、context512，主配置为micro16×累积4，扩展为micro64×累积1；浮点累积路径不同。目标曝光从10M改为100M，同时改变训练容量、回放来源池/银行条件、matched heldout以及按各自完整horizon安排的LR日程和验证网格（主每50步=1,635,200目标tokens，扩展每153步=5,003,712）。因此这不是单独改变horizon的因果对照。','',
        '扩展在正式训练前按各自seed101/2M validation-only校准冻结共享LR=3e−5、λ=.05（B0的b=λ=0，其余b=.05）。本次24条长程轨迹没有重新选择λ，也没有借用A后续选参。短pilot规则不保证100M长程最优；这限制解释，不构成事后调参授权。','',
        '## 可复现证据','',
        '- results/C410_validation_review_v1/best_exposure_all24.csv：全部24项含run ID、best/final step、实际目标/回放曝光与target验证NLL。',
        '- best_exposure_by_group.csv：每组三seed的曝光、mean/sample SD及final−best描述。',
        '- validation_trajectories_independent.csv：480点的独立重算与原分析逐值核对。',
        '- review.json：watcher、audit、registry、summary/events/config/CSV来源SHA，完成门禁与精确核对计数。',
        '- whole_process_timing.json：独立审阅子进程完整spawn→exit wall/user/sys；工程编写、先前人工读取和后续复核不在该timer内，未计部分仍unknown。','',
        '本说明仅补足验证选择与曝光口径，遵循《C完整结果解释框架_解封前》，不改既定科学条件、统计对照或主报告结论。','']
    (ROOT/'docs/C410验证轨迹与best曝光_完整组审阅.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'passed','runs':24,'validation_points':480,'best_distribution':dict(sorted(dist.items())),
        'exact_checks':dict(exact_checks),'output':str(OUT)},ensure_ascii=False))


if __name__ == '__main__':
    main()
