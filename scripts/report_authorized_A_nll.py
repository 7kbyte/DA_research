#!/usr/bin/env python3
"""Display already-complete authorized A tables; no scorer or new selection."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from replay_study import supplemental_analysis_guard_v2 as gate


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join('---' for _ in headers)+'|',*['|'+'|'.join(map(str,r))+'|' for r in rows]])


def report(watch_proof,output):
    v=gate.Verifier();watch=v.proof(watch_proof)
    if watch.get('status')!='completed' or watch.get('scope')!='ONLY_A_NLL_full126_and_validation_selected18':raise PermissionError('Complete explicitly authorized A watcher receipt required before any score table is opened')
    complete=v.proof(watch['analysis_complete']);plan=v.proof(complete['plan']);authority=v.proof(complete['authority'])
    readiness=v.proof(complete['readiness']);first=v.proof(complete['first_read_receipt'])
    if (plan['contract_id']!='supplement_A_nll' or authority.get('allow_score_analysis') is not True or authority['plan']!=complete['plan'] or
            readiness.get('ready') is not True or not readiness.get('all_source_baseline_alias_output_integrity_verified') or
            first['plan']!=complete['plan'] or first['authority']!=complete['authority'] or not first.get('before_first_score_parse')):
        raise PermissionError('Full A scope/authority/first-read provenance differs')
    root=Path(watch['analysis_complete']['path']).parent;aux=root/'validation_selected_policy'
    def rows(path):
        v.bind(path)
        with Path(path).open() as f:return list(csv.DictReader(f))
    endpoints=rows(root/'final/endpoints.csv');selected=rows(aux/'selected_endpoints.csv');stats=rows(root/'final/endpoint_statistics.csv')
    tuned=rows(aux/'selected_policy_statistics.csv');same=rows(root/'final/paired_statistics.csv')
    training=rows(root/'final/actual_training_costs.csv');inference=rows(root/'final/physical_inference_costs.csv')
    intervals=rows(aux/'selected_document_bootstrap.csv')
    if len(endpoints)!=126 or len(selected)!=18 or {r['run_id'] for r in selected}!=set(plan['selection']['selected_ids']):raise ValueError('Full126/selected18 table identity differs')
    if len({(r['group'],r['weight']) for r in endpoints})!=42 or len({r['comparison'] for r in tuned})!=6:raise ValueError('Incomplete lambda grid or six-policy contrasts')
    manifest=v.proof(plan['manifest']);state=gate.read(ROOT/'logs/supplemental_job_state.json')
    logical={r['run_id']:r for r in manifest['records']}
    for r in selected:
        source=logical[r['run_id']]
        for k in ('group','actual_run_id','checkpoint_sha256'):
            if r[k]!=str(source[k]):raise ValueError('Selected table/source differs: '+k)
        if int(r['target_prediction_tokens'])!=source['target_prediction_tokens_at_checkpoint'] or int(r['replay_prediction_tokens'])!=source['replay_prediction_tokens_at_checkpoint']:raise ValueError('Selected exposure differs')
    lookup={(r['group'],float(r['weight']),r['metric']):r for r in stats}
    tuned_lookup={(r['comparison'],r['metric']):r for r in tuned}
    choices={c['group']:c['weight'] for c in plan['selection']['choices']};groups=('B1','B2','B3','B4','M','C')
    metrics=('g_new','f_old_c4','f_old_wikitext')
    fmt=lambda r:f"{float(r['mean']):.4f} ± {float(r['std_ddof1']):.4f}"
    selected_mean=[[g,choices[g],*[fmt(lookup[g,float(choices[g]),metric]) for metric in metrics]] for g in groups]
    individuals=[[r['group'],r['weight'],r['training_seed'],*[f'{float(r[m]):.5f}' for m in metrics]] for r in sorted(selected,key=lambda r:(groups.index(r['group']),int(r['training_seed'])))]
    pairs=[]
    for name in ('M-B4','C-B4','B2-B1','B4-B3','B3-B1','B4-B2'):
        a,b=name.split('-');pairs.append([name,f'{choices[a]}/{choices[b]}',*[fmt(tuned_lookup[name,metric]) for metric in metrics]])
    cost={'unique_physical_training_runs':len(training),'incremental_new_training_runs':sum(float(r['incremental_phase_training_cost_factor'])==1 for r in training),
        'full126_training_committed_parent_wall_sum_seconds':sum(float(r['committed_parent_sum_seconds']) for r in training),
        'incremental108_training_committed_parent_wall_sum_seconds':sum(float(r['incremental_phase_parent_sum_seconds']) for r in training),
        'training_parent_sums_are_parallel_run_wall_not_study_elapsed_or_pure_GPU_seconds':True,
        'A_NLL_incremental_scorer_subprocess_wall_sum_seconds':sum(float(r['incremental_inference_subprocess_wall_seconds']) for r in inference),
        'A_NLL_whole_scheduler_slot_sum_seconds':sum(state['jobs'][j['id']]['elapsed_seconds'] for j in manifest['jobs']),
        'A_NLL_newforward_scheduler_slot_sum_seconds':sum(state['jobs'][j['id']]['elapsed_seconds'] for j in manifest['jobs'] if j['kind']!='reuse'),
        'A_NLL_reuse_materialization_scheduler_slot_sum_seconds':sum(state['jobs'][j['id']]['elapsed_seconds'] for j in manifest['jobs'] if j['kind']=='reuse'),
        'scorer_wall_nested_in_scheduler_wall_do_not_add':True,'teacher_subtimers_nested_do_not_add':True,
        'CPU_full_guard_statistics_plots_wall_seconds':watch['analysis_wall_seconds'],
        'guard_CPU_IO_wall_nested_in_CPU_analysis':readiness['score_blind_verification_wall_seconds'],
        'new_generation_or_bank_preparation_for_A':False,'original_pools_banks_shared_cost_not_repeated_per_lambda':True,
        'coverage':'Committed-path training plus successful A inference/analysis; global inventory/telemetry remains authoritative for pilots, failures, uncommitted work, downloads, archives and publication'}
    if cost['unique_physical_training_runs']!=126 or cost['incremental_new_training_runs']!=108:raise ValueError('Training alias cost inventory differs')
    paragraphs=[
      '# 补充A：完整NLL网格与验证所选策略',
      '完整126端点、全部145项评分依赖闭包通过来源/输出/attestation核验后，已在root显式授权下分析。A中的18个主实验复用结果此前已见；新增108个端点没有用于修改已冻结的训练、λ选择或其他补充条件。行为、B、C分数继续封存。',
      '六方法各7个λ×3训练seed；全部为final端点。所选18点仅按已经冻结的验证选择ID连接，不新增推理、不用test重新选λ。每点目标预测token为10,000,000、回放预测token为500,000；完整训练成本与选点曝光分开保留。',
      '## 各方法验证所选λ：三seed均值±样本SD',
      table(['方法','所选λ','G_new','F_C4','F_Wiki'],selected_mean),
      'G_new为initial−adapted目标NLL，越大越好；F为adapted−initial通用域NLL，越小越好。此处只描述NLL，不能替代行为能力或总体保留结论。全部原始NLL/KL及完整小数见CSV。',
      '## 所选18点个体值',table(['方法','λ','训练seed','G_new','F_C4','F_Wiki'],individuals),
      '## 各自验证选λ后的策略比较',
      table(['左−右','λ左/右','ΔG_new','ΔF_C4','ΔF_Wiki'],pairs),
      '逐相同训练seed先相减，再汇总三个个体的均值与样本SD。两边method及所选λ共同变化，不能解释成固定同λ的objective因果效应。三seed也参与共享λ选择，因此此SD条件于该选择，不估计重复整个选参流程的不确定性。',
      '## 完整同λ对照与条件区间',
      f'同λ结果保留全部7λ×6预定对照，没有按测试结果筛选某个共同λ；原始paired_effects/paired_statistics与document_bootstrap独立保存。所选策略另有{len(intervals)}个逐seed×域×NLL/KL的配对文档bootstrap区间（5000次，固定seed20261106）。',
      '区间重采样单位是同一文档及其全部chunks，条件于已训练checkpoint；不是跨训练seed的置信区间，也不自动代表独立目标总体。表内原始区间为左−右NLL/KL；目标增益差须反转目标NLL区间符号。条件区间另表，不与三seed SD混画。',
      '## 全126曲线',f'![全126端点；每λ三个seed点与均值±样本SD]({root}/full_lambda_curves/full_seven_lambda_curves.png)',
      '图为7×6.6in，配套CSV及PNG/PDF/SVG；完整七λ与全部seed点保留。',
      '## 成本与曝光',
      f"126个已存在物理训练合计committed父wall {cost['full126_training_committed_parent_wall_sum_seconds']:.2f}s；其中本阶段新增108个训练 {cost['incremental108_training_committed_parent_wall_sum_seconds']:.2f}s，主18项不重复记为新增。并行run wall之和不是研究elapsed或纯GPU时间。",
      f"A-NLL scorer子进程wall合计 {cost['A_NLL_incremental_scorer_subprocess_wall_sum_seconds']:.2f}s，已包含teacher计算；126评分/复用作业的whole-slot合计 {cost['A_NLL_whole_scheduler_slot_sum_seconds']:.2f}s（含复用materialization {cost['A_NLL_reuse_materialization_scheduler_slot_sum_seconds']:.2f}s）。子进程wall嵌在whole-slot内，不相加。",
      f"本次完整CPU核验、统计和出图wall {cost['CPU_full_guard_statistics_plots_wall_seconds']:.2f}s，内含guard CPU/I/O {cost['guard_CPU_IO_wall_nested_in_CPU_analysis']:.2f}s。checkpoint命中是历史完整读取receipt的复用，本次fresh字节与historical covered bytes分列于readiness，不能把命中写成本次全读。",
      'A没有新生成pool/bank，原共享准备费不能乘以七λ重复计入。此处是committed-path/成功评分账目，不覆盖回滚或丢失工作；全研究inventory和GPU遥测另列，未测下载/预处理/归档/报告等不按0处理。',
      '## 可追溯文件',
      f'- 全126端点与同λ配对：{root}/final/\n- 所选18、六组策略配对及条件区间：{aux}/\n- 曲线及完整精度CSV：{root}/full_lambda_curves/\n- 首次解封回执：{complete["first_read_receipt"]["path"]}\n- 计划SHA：{complete["plan"]["sha256"]}\n- 完整验证λ选择SHA：{plan["selection"]["selection"]["sha256"]}']
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    v.finish();reportproof=gate.immutable(output.with_suffix('.provenance.json'),{'watch_completion':watch_proof,'analysis_complete':watch['analysis_complete'],'first_read':complete['first_read_receipt'],
        'plan':complete['plan'],'authority':complete['authority'],'analysis_inputs_sha256':v.sources,'cost_summary':cost,'report_source':gate.proof(__file__),
        'new_selection_or_model_inference':False,'reported_scope':'ONLY complete A-NLL full126/selected18','same_lambda_comparison_cells':len({(r['comparison'],r['weight']) for r in same})})
    with output.open('x') as f:f.write('\n\n'.join(paragraphs)+'\n')
    return {'report':gate.proof(output),'provenance':reportproof,'cost_summary':cost}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch-complete',type=Path,required=True);p.add_argument('--watch-complete-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(report({'path':str(a.watch_complete.resolve()),'sha256':a.watch_complete_sha256},a.output),ensure_ascii=False))
