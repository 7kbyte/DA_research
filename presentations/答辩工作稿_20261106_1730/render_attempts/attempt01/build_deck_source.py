#!/usr/bin/env python3
"""CPU-only editable defense working draft from explicitly opened audited CSVs."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import html
import json
import os
import shutil
import statistics as st
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = [('g_new', 'G new ↑'), ('f_old_c4', 'F C4 ↓'), ('f_old_wikitext', 'F WikiText ↓')]
GROUPS = ['B0', 'B1', 'B2', 'B3', 'B4', 'D', 'M', 'C']
COLORS = dict(zip(GROUPS, ['#556270', '#d17a22', '#256a9b', '#b95663', '#007d78', '#598f8b', '#7761a4', '#917323']))
MARKERS = ['o', 's', '^']
SOURCES = {
    'primary_final': 'results/analysis/primary_nll_v2/final/endpoints.csv',
    'primary_best': 'results/analysis/primary_nll_v2/best/endpoints.csv',
    'A_final': 'results/supplemental_analysis_v2/supplement_A_nll/final/endpoints.csv',
    'A_selected': 'results/supplemental_analysis_v2/supplement_A_nll/validation_selected_policy/selected_endpoints.csv',
    'B_final': 'results/supplement_B_test_review_v2/all_60_endpoints.csv',
    'C410_all': 'results/C410_nll_independent_review_v1/endpoints_independent.csv',
    'primary_cost': 'results/analysis/primary_nll_v2/final/actual_training_costs.csv',
    'outline': 'docs/论文写作与答辩提纲.md',
    'primary_discussion': 'docs/主NLL测试结果讨论.md',
    'A_discussion': 'docs/补充A_NLL完整结果.md',
    'B_discussion': 'docs/补充B_完整60run测试结果解释.md',
    'C410_discussion': 'docs/C410_100M_NLL完整结果与正文素材.md',
    'primary_final_manifest': 'results/analysis/primary_nll_v2/final/analysis_manifest.json',
    'primary_best_manifest': 'results/analysis/primary_nll_v2/best/analysis_manifest.json',
    'A_complete': 'results/supplemental_analysis_v2/supplement_A_nll/COMPLETE.json',
    'B_complete': 'results/supplemental_analysis_v2/supplement_B_nll/COMPLETE.json',
    'C410_complete': 'results/C410_nll_independent_review_v1/REPORT_COMPLETE.json',
}
stages = []
source_evidence = {}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stage(name, wall, cpu):
    stages.append(dict(name=name, wall_seconds=time.perf_counter()-wall,
                       process_CPU_seconds=time.process_time()-cpu,
                       semantics='Nonoverlapping child stage; nested in whole build child.'))


def load_sources():
    for folder in ['data', 'sources', 'figures', 'raster', 'fonts']:
        (HERE/folder).mkdir(exist_ok=True)
    values = {}
    for name, relative in SOURCES.items():
        p = ROOT/relative
        raw = p.read_bytes()
        dest = HERE/('data' if p.suffix == '.csv' else 'sources')/(name+p.suffix)
        dest.write_bytes(raw)
        source_evidence[name] = dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(),
                                    bytes=len(raw), retained_copy=str(dest))
        if p.suffix == '.csv':
            values[name] = list(csv.DictReader(raw.decode().splitlines()))
    font = ROOT/'artifacts/reporting/NotoSerifCJKsc-Regular.otf'
    shutil.copyfile(font, HERE/'fonts'/font.name)
    source_evidence['font'] = dict(path=str(font), sha256=digest(font), bytes=font.stat().st_size)
    assert [len(values[k]) for k in ['primary_final', 'primary_best', 'A_final', 'A_selected', 'B_final', 'C410_all']] == [36,36,126,18,60,48]
    for name in ['primary_final', 'primary_best', 'A_final', 'A_selected', 'B_final']:
        assert {r['data_family'] for r in values[name]} == {'main_v3'}
    for name in ['primary_final', 'primary_best', 'A_final', 'A_selected', 'B_final', 'C410_all']:
        assert {r['model_tag'] for r in values[name]} == {'pythia-410m'}
        assert all(np.isfinite(float(r[m])) for r in values[name] for m,_ in METRICS)
    assert {r['data_family'] for r in values['C410_all']} == {'extended_v1'}
    assert {r['phase_name'] for r in values['C410_all']} == {'extended_pythia_410m_100m'}
    assert {r['training_seed'] for r in values['B_final']} == {'11'}
    assert {r['generation_seed'] for r in values['B_final']} == {'11'}
    assert {r['selection_seed'] for r in values['B_final']} == {str(x) for x in range(1001,1011)}
    return values


plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':20, 'axes.titlesize':23,
                     'axes.labelsize':21, 'xtick.labelsize':18, 'ytick.labelsize':21,
                     'svg.fonttype':'none', 'pdf.fonttype':42, 'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.spines.left':False})
plot_evidence = []


def save_figure(fig, name, rows, extra=None):
    fig.savefig(HERE/'figures'/(name+'.svg'), facecolor='white')
    fig.savefig(HERE/'figures'/(name+'.png'), dpi=140, facecolor='white')
    fig.savefig(HERE/'figures'/(name+'.pdf'), facecolor='white')
    plt.close(fig)
    plot_evidence.append(dict(figure=name, logical_endpoint_points_per_metric=len(rows),
                             metrics=[m for m,_ in METRICS], all_input_rows_plotted=True,
                             rows=list(r['record_id'] for r in rows), **(extra or {})))


def dot_plot(rows, name, labels, key, common_limits=None, draw=False):
    fig, axes = plt.subplots(1,3,figsize=(15,5.15))
    fig.subplots_adjust(left=.19,right=.985,bottom=.21,top=.9,wspace=.20)
    seeds = list(range(1001,1011)) if draw else [11,22,33]
    palette = plt.cm.tab10(np.arange(10))
    for ax,(metric,title) in zip(axes,METRICS):
        allvals=[]
        for j,(category,label) in enumerate(labels):
            rr = [r for r in rows if key(r)==category]
            assert len(rr)==len(seeds), (name,category,len(rr))
            vals=[float(r[metric]) for r in rr];allvals+=vals
            for r in rr:
                s=int(r['selection_seed'] if draw else r['training_seed']);i=seeds.index(s)
                offset=(i-(len(seeds)-1)/2)*(.042 if draw else .12)
                ax.scatter(float(r[metric]),j+offset,s=35 if draw else 65,
                           marker='o' if draw else MARKERS[i],
                           color=palette[i] if draw else COLORS[r['group']],alpha=.9,zorder=3)
            ax.errorbar(st.mean(vals),j,xerr=st.stdev(vals),fmt='D',markersize=6,
                        color='#202a34',linewidth=1.8,capsize=3,zorder=4)
        ax.axvline(0,color='#909aa3',linewidth=.9,linestyle='--')
        ax.set_title(title,pad=12);ax.set_ylim(len(labels)-.45,-.55)
        ax.set_yticks(range(len(labels)));ax.grid(axis='x',color='#e5e8ec',zorder=0)
        ax.tick_params(axis='y',length=0)
        if ax is axes[0]:ax.set_yticklabels([x[1] for x in labels])
        else:ax.set_yticklabels([])
        if common_limits is not None:ax.set_xlim(common_limits[metric])
        else:
            lo=min(allvals+[0]);hi=max(allvals+[0]);pad=max((hi-lo)*.10,.002)
            ax.set_xlim(lo-pad,hi+pad)
        ax.locator_params(axis='x',nbins=4)
        ax.set_xlabel('nats / prediction token',labelpad=10,fontsize=17)
    if draw:
        handles=[Line2D([],[],marker='o',linestyle='',color=palette[i],label=str(s)) for i,s in enumerate(seeds)]
        fig.legend(handles=handles,loc='lower center',ncol=10,frameon=False,fontsize=12.5,
                   handletextpad=.2,columnspacing=.7,bbox_to_anchor=(.59,0))
    else:
        handles=[Line2D([],[],marker=MARKERS[i],linestyle='',color='#536371',label=f'seed {s}') for i,s in enumerate(seeds)]
        handles.append(Line2D([],[],marker='D',color='#202a34',label='mean ± sample SD'))
        fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=17,bbox_to_anchor=(.59,0))
    save_figure(fig,name,rows,dict(shared_final_best_axis_limits=common_limits,replicate_unit='selection draw' if draw else 'pipeline seed'))


def A_curves(rows, selected):
    fig,axes=plt.subplots(1,3,figsize=(15,5.15));fig.subplots_adjust(left=.065,right=.985,bottom=.24,top=.88,wspace=.30)
    groups=['B1','B2','B3','B4','M','C'];lambdas=sorted({float(r['weight']) for r in rows});assert len(lambdas)==7
    for ax,(metric,title) in zip(axes,METRICS):
        for g in groups:
            means=[]
            for lam in lambdas:
                rr=[r for r in rows if r['group']==g and float(r['weight'])==lam];assert len(rr)==3
                vals=[float(r[metric]) for r in rr];means.append(st.mean(vals))
                for r in rr:
                    i=[11,22,33].index(int(r['training_seed']))
                    ax.scatter(lam*(1+(.035*(i-1))),float(r[metric]),marker=MARKERS[i],s=18,color=COLORS[g],alpha=.65,zorder=2)
                ax.errorbar(lam,st.mean(vals),yerr=st.stdev(vals),color=COLORS[g],linewidth=1,capsize=2,alpha=.65)
            ax.plot(lambdas,means,color=COLORS[g],linewidth=2,label=g)
            sr=[r for r in selected if r['group']==g];assert len(sr)==3
            ax.scatter(float(sr[0]['weight']),st.mean(float(r[metric]) for r in sr),marker='*',s=125,color=COLORS[g],edgecolor='white',linewidth=.5,zorder=5)
        ax.set_xscale('log');ax.set_xticks(lambdas);ax.set_xticklabels([f'{x:g}' for x in lambdas],rotation=45,ha='right',fontsize=15)
        ax.set_title(title,pad=12);ax.grid(alpha=.22);ax.axhline(0,color='#84909b',linestyle='--',linewidth=.7);ax.set_xlabel('replay weight λ',fontsize=20)
    handles=[Line2D([],[],color=COLORS[g],linewidth=3,label=g) for g in groups]
    handles.append(Line2D([],[],color='#263746',marker='*',linestyle='',markersize=11,label='validation-selected λ'))
    fig.legend(handles=handles,loc='lower center',ncol=7,frameon=False,fontsize=15,bbox_to_anchor=(.52,0))
    save_figure(fig,'A_all_126',rows,dict(all_lambdas=lambdas,selected_record_ids=[r['record_id'] for r in selected],selection='frozen validation IDs; no test reselection'))


def section(number,title,body,source,cls=''):
    return f'<section class="slide {cls}"><div class="kicker">毕业设计 · 答辩演示工作稿 / 截至 2026-11-06 17:30 UTC</div><h1>{title}</h1>{body}<footer><span>{source}</span><b>{number:02d} / 12</b></footer></section>'


def figure(name):return f'<img class="chart" src="figures/{name}.svg" alt="全部既定端点三指标；点为个体，菱形及误差线为均值和样本SD">'
def card(title,body):return f'<div class="card"><h2>{title}</h2><p>{body}</p></div>'


def write_slides(values):
    primary=values['primary_final'];best=values['primary_best'];A=values['A_final'];B=values['B_final'];C=values['C410_all']
    mainlabels=[((g,0.0 if g=='B0' else .05),f'{g}  b={0 if g=="B0" else .05:g}') for g in GROUPS]
    mainlabels += [((g,b),f'{g}  b={b:g}') for g in ['B4','M'] for b in [.01,.1]]
    primarykey=lambda r:(r['group'],float(r['ratio']))
    dot_plot(primary,'primary_all36_final',mainlabels,primarykey)
    dot_plot(best,'primary_all36_best',mainlabels,primarykey)
    A_curves(A,values['A_selected'])
    blabels=[((g,k),f'{g}  K={k}') for k in [20,196] for g in ['B4','M','C']]
    dot_plot(B,'B_all60_final',blabels,lambda r:(r['group'],int(r['bank_chunks'])),draw=True)
    clim={}
    for metric,_ in METRICS:
        vv=[float(r[metric]) for r in C]+[0];pad=(max(vv)-min(vv))*.075
        clim[metric]=(min(vv)-pad,max(vv)+pad)
    clabels=[(g,g) for g in GROUPS]
    dot_plot([r for r in C if r['checkpoint']=='final'],'C410_all24_final',clabels,lambda r:r['group'],clim)
    dot_plot([r for r in C if r['checkpoint']=='best'],'C410_all24_best',clabels,lambda r:r['group'],clim)
    cost=sum(float(r['committed_parent_sum_seconds']) for r in values['primary_cost'])
    slides=[]
    slides.append(section(1,'有限回放库下，适应与遗忘如何取舍？',
        '<p class="lead">Pythia-410M 在英文数学文本上继续训练：<br>一次控制比较，而非新回放算法。</p><div class="cards three">'+
        card('来源','替代通用文本与自生成文本，哪一种更合适？')+card('目标','硬标签交叉熵与冻结教师KL，有何差别？')+card('选择','有限库里，分层与聚类是否稳定胜出？')+
        '</div><p class="takeaway">已开放：主实验、A权重网格、B选库draw、C410长程NLL。<br>数学文本NLL不是数学推理正确率。</p>', '来源：论文写作与答辩提纲；完整NLL结果材料', 'cover'))
    table=''.join(f'<tr><td><b>{g}</b></td><td>{src}</td><td>{obj}</td><td>{sel}</td></tr>' for g,src,obj,sel in [
        ('B0','无回放','仅目标CE','—'),('B1','C4替代通用文本','CE','random'),('B2','C4替代通用文本','教师→学生KL','random'),
        ('B3','冻结模型自生成','CE','random'),('B4','冻结模型自生成','教师→学生KL','random'),('D','同一生成池','教师→学生KL','整块精确去重后random'),
        ('M','同一生成池','教师→学生KL','teacher-NLL五分位分层'),('C','同一生成池','教师→学生KL','特征聚类代表')])
    slides.append(section(2,'八组设计：先分开来源、目标和选择',f'<table class="design"><thead><tr><th>组</th><th>回放来源</th><th>回放目标</th><th>选库规则</th></tr></thead><tbody>{table}</tbody></table><p class="note">主设置：seed 11/22/33；其余控制共享。D与B4本次bank和完整权重相同，属于null对照；D不增加独立重复。</p>', '来源：主冻结协议、主NLL/C410完整结果；C组≠补充C阶段'))
    slides.append(section(3,'K、b与λ控制的是三件不同的事',
        '<div class="cards three">'+card('库容量 K','主库196块 × 511预测位置<br><b>100,156个位置</b><br>不是累计训练曝光。')+card('累计曝光 b','目标10M，回放0.5M<br><b>b = Tᵣ / Tₜ = .05</b><br>总处理占比p≈4.76%。')+card('损失权重 λ','两项loss各自求均值<br><b>L = CEₜ + λ Lᵣ</b><br>主λ=.05，独立于b。')+
        '</div><p class="equation">单预测token相对系数约 λ/b；<br>它不是梯度、Adam更新或计算量之比。</p><p class="note">0.5M回放≈同一196块库的4.99次容量曝光。A改变λ，B改变K；不把重复曝光叫新增独立数据。</p>', '来源：主36运行预算口径；方法推导与解释边界'))
    slides.append(section(4,'指标方向与三种不确定性',
        '<div class="cards three">'+card('G new ↑','initial数学NLL − adapted数学NLL<br>越大越好。')+card('F C4 ↓','adapted C4 NLL − initial<br>越小越好，负值才是改善。')+card('F WikiText ↓','另一旧域代理，单独报告<br>C4并非Pythia原预训练集。')+
        '</div><div class="two"><div><h2>图中：全部个体 + mean ± SD</h2><p>主/A/C410：3个pipeline seed。<br>B：固定pool/train=11的10个选库draw。</p></div><div><h2>文档bootstrap：另一个问题</h2><p>5000次配对文档重采样，条件于固定checkpoint；不能代替seed或draw的SD。</p></div></div><p class="note">本稿图固定token加权。等文档权重完整保留：例如C主final的G均值为−.00357（token）/+.01832（等文档），估计对象不同，不能择优报告。</p>', '来源：主/A/B/C410完整表；单位均为nats / prediction token'))
    slides.append(section(5,'主实验：固定10M final，KL减少旧域退化',
        figure('primary_all36_final')+'<p class="takeaway compact">完整36点 × 三指标；同来源KL相对CE的旧域损失更小。<br>生成来源与M/C没有统一优势；D = B4，仍保留两组标签。</p>', '来源：primary_nll_v2/final/endpoints.csv；λ=.05；全部3 seed，误差线为样本SD'))
    slides.append(section(6,'主实验best：验证选点改善，但不是同预算替代',
        figure('primary_all36_best')+'<p class="takeaway compact">36个逻辑配置：best的G更高、两个F更低（描述计数）。<br>best实际目标曝光1.6352M–4.9056M；每条完整训练仍为10M。</p>', '来源：primary_nll_v2/best/endpoints.csv；只有两个test观测端点，不推断全程单调'))
    slides.append(section(7,'A：结论对λ敏感，必须保留全部七λ',
        figure('A_all_126')+'<p class="takeaway compact">126逻辑条件＝108新增训练 + 18主实验复用；每点3 seed。<br>验证所选λ：B1 .05｜B2 1｜B3 .1｜B4 5｜M 5｜C 1。</p><p class="note small">星号只连接既有验证选择ID；测试不重选λ。各自选λ比较的是“方法＋选参政策”，不是固定同λ的objective效应。</p>', '来源：A完整126端点、冻结验证选择18端点；全部候选及个体均显示'))
    slides.append(section(8,'B：组成更稳，不保证训练终点波动更小',
        figure('B_all60_final')+'<p class="takeaway compact">60端点全部显示；n=10是selection draw，不是训练seed。<br>M在K196三项均值较有利，K20三项均值反向；六格端点SD未低于B4。</p>', '来源：B完整60端点；pool/train=11，λ=b=.05；颜色对应1001–1010'))
    slides.append(section(9,'C410：100M final，保留收益仍依赖来源',
        figure('C410_all24_final')+'<p class="takeaway compact">24端点，matched extended initial；与下一页共用坐标范围。<br>B1的大遗忘完整保留；生成KL相对CE的数学ΔG为−.00684。</p>', '来源：C410独立核验48端点；final=100M；λ=b=.05；D/B4三对全状态相等'))
    slides.append(section(10,'C410 best：端点会改变数学增益差的方向',
        figure('C410_all24_best')+'<p class="takeaway compact">生成KL−CE的数学ΔG为+.01441；24项best六指标均优于各自final。<br>实际best约35.026M / 40.030M / 55.041M；完整训练均为100M。</p>', '来源：C410完整端点与曝光表；相关数据/日程扩展，不是纯horizon因果实验'))
    costrows=''.join(f'<tr><td>{name}</td><td>{n}</td><td>{secs/3600:.2f} h</td></tr>' for name,n,secs in [
        ('主实验','36',cost),('A新增','108（另18复用）',36992.93),('B','60',20520.290),('C410','24',50214.167753)])
    slides.append(section(11,'贡献在受控比较与边界，不在宣称普遍胜出',
        '<div class="two"><div><h2>可以支持</h2><ul><li>分开来源、objective、选择与预算。</li><li>完整保留负结果、个体与配对。</li><li>揭示λ、K和端点的条件依赖。</li></ul></div><div><h2>训练已提交路径父项</h2><table class="cost"><tr><th>阶段</th><th>独立训练</th><th>累计parent wall</th></tr>'+costrows+'</table></div></div><p class="note">表中是各run父项累计，不是研究elapsed或纯GPU秒；teacher已内嵌，final/best不重复收费。共享生成/特征费计一次；下载、归档等不在此训练表内。</p><p class="takeaway">局限：替代旧域、有限pool与文档抽样；NLL分层不等于梯度分层，文本NLL不等于推理能力。</p>', '来源：主/A/B/C410已审成本表；完整项目账本另列，未计工作不能补0'))
    slides.append(section(12,'截至17:30：哪些结论已完成，哪些还不能说',
        '<div class="status"><div><span class="badge">已开放</span><h2>主 / A / B / C410 NLL</h2><p>完整比较、全部既定条件与两种文档权重。<br>本稿只展示token三指标；完整表和条件文档区间可追溯。</p></div><div><span class="badge pending">待完成／解封</span><h2>1B、2.8B与行为评价</h2><p>不预写规模趋势，不用部分运行替代完整模型阶段；<br>行为分数仍封存，不宣称能力保留或推理提升。</p></div></div><p class="lead closing">答辩核心回答：<br>KL减少本设置的旧域退化；<br>来源与选库没有跨条件稳定赢家。</p><p class="note small">完整来源：论文写作与答辩提纲、主NLL讨论、A/B/C410结果材料；文献全文支持范围见literature/references.bib。作者/学校信息与最终完成页由学生后续补齐。</p>', '工作稿，不替代最终学位论文；工程SHA、逐点CSV与渲染QA见相邻receipt'))
    css='''@font-face{font-family:NotoCJK;src:url(fonts/NotoSerifCJKsc-Regular.otf)}
    @page{size:16in 9in;margin:0}*{box-sizing:border-box}body{margin:0;background:#fff;color:#172c3e;font-family:NotoCJK,serif;font-size:25pt}
    .slide{position:relative;width:16in;height:9in;padding:.38in .6in .68in;break-after:page;background:white}.slide:last-child{break-after:auto}
    .kicker{font-size:12pt;color:#617887;letter-spacing:.5px;margin:0 0 .10in}h1{font-size:34pt;line-height:1.3;margin:0 0 .20in;font-weight:bold;letter-spacing:-.3px}
    h2{font-size:24pt;line-height:1.35;margin:.06in 0 .13in;color:#087c78}p{line-height:1.48;margin:.12in 0}b{font-weight:bold}
    .lead{font-size:31pt;line-height:1.5;margin:.2in 0}.cover h1{font-size:41pt;margin-top:.32in}.cards{display:flex;gap:.18in;margin:.18in 0}.card{background:#eff5f7;border-top:5px solid #087c78;padding:.20in .22in;flex:1}.card p{font-size:23pt}.three .card{width:33.33%}
    .takeaway{padding:.14in .20in;border-left:5px solid #d49d33;background:#fbf5e8;font-size:25pt;line-height:1.5;margin-top:.18in}.compact{font-size:22pt;margin-top:0;padding:.08in .18in;line-height:1.42}
    .note{font-size:19pt;line-height:1.45;color:#486171;margin:.16in 0}.small{font-size:16pt}.equation{font-size:31pt;text-align:center;line-height:1.55;margin:.24in 0}
    table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:.07in .12in;border-bottom:1px solid #d9e4e8;line-height:1.25}th{background:#eaf1f4;color:#345769}table.design{font-size:22pt}.design td:first-child{width:.65in}.two{display:flex;gap:.45in;margin:.16in 0}.two>div{width:50%}.two p{font-size:23pt}ul{padding-left:.36in;line-height:1.6}li{margin:.10in 0}table.cost{font-size:18pt}.chart{display:block;width:14.8in;height:5.08in;object-fit:contain;margin:-.03in 0 0 -.05in}
    footer{position:absolute;left:.6in;right:.6in;bottom:.22in;border-top:1px solid #d5dfe5;padding-top:.08in;display:flex;justify-content:space-between;font-size:10.7pt;color:#6b7b87}footer span{max-width:93%}footer b{color:#17384c}
    .status{display:flex;gap:.24in}.status>div{width:50%;background:#f0f5f7;padding:.24in}.status p{font-size:22pt}.badge{font-size:16pt;background:#007d78;color:white;padding:.03in .14in}.pending{background:#8b7136}.closing{font-size:28pt;margin:.24in 0 .12in}
    '''
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>答辩演示工作稿 · 截至17:30</title><style>'+css+'</style><body>'+''.join(slides)+'</body></html>'
    (HERE/'答辩演示工作稿.html').write_text(document,encoding='utf-8')
    notes=['# 答辩演示工作稿（可编辑HTML源＋讲稿索引）','截至2026-11-06 17:30 UTC；12页，16:9。HTML是可直接编辑的排版源，build_deck.py可从相邻data重画图。数字冻结在该时间口径，不按后来测试改叙述。']
    for i,s in enumerate(slides,1):
        import re
        title=re.search(r'<h1>(.*?)</h1>',s).group(1)
        notes.append(f'\n## {i}. {title}\n\n'+re.sub('<[^>]+>',' ',s).replace('  ',' '))
    notes.append('\n## 完整结果链接\n\n- 主结果与两种权重：[讨论](../../docs/主NLL测试结果讨论.md)\n- A全部λ及所选政策：[完整结果](../../docs/补充A_NLL完整结果.md)\n- B全部draw与两种权重：[完整结果](../../docs/补充B_完整60run测试结果解释.md)\n- C410全端点：[完整结果](../../docs/C410_100M_NLL完整结果与正文素材.md)\n- 写作与问答：[提纲](../../docs/论文写作与答辩提纲.md)\n\n图中全部seed/draw身份、所有λ/方法/预算均保留在data，不以均值或表现删除端点。D/B4重合明确保留null身份。教师KL恒等式只针对固定学生/前缀的教师标签期望，不把同bank训练后的经验CE当KL。')
    (HERE/'讲稿与来源.md').write_text('\n'.join(notes),encoding='utf-8')


def render_and_inspect():
    from weasyprint import HTML
    import pypdfium2 as pdfium
    from PIL import Image,ImageOps,ImageDraw
    doc=HTML(filename=str(HERE/'答辩演示工作稿.html')).render()
    assert len(doc.pages)==12, f'Expected12pages, actual{len(doc.pages)}'
    outliers=[]
    for i,page in enumerate(doc.pages,1):
        for box in page._page_box.descendants():
            if type(box).__name__=='TextBox' and getattr(box,'text','').strip():
                x,y=box.position_x,box.position_y;w,h=box.width,box.height
                if x<0 or y<0 or x+w>page.width+1 or y+h>page.height+1:
                    outliers.append(dict(page=i,text=box.text,x=x,y=y,width=w,height=h))
    assert not outliers, outliers
    doc.write_pdf(HERE/'答辩演示工作稿.pdf')
    pdf=pdfium.PdfDocument(str(HERE/'答辩演示工作稿.pdf'));assert len(pdf)==12
    contact=Image.new('RGB',(1920,4*385),'#dce3e8');draw=ImageDraw.Draw(contact)
    pages=[]
    for i in range(len(pdf)):
        page=pdf[i];w,h=page.get_size();assert abs(w/h-16/9)<1e-8
        image=page.render(scale=1600/w).to_pil().convert('RGB')
        path=HERE/'raster'/f'page_{i+1:02d}.png';image.save(path)
        thumb=ImageOps.contain(image,(630,355));x=(i%3)*640+(640-thumb.width)//2;y=(i//3)*385+22
        contact.paste(thumb,(x,y));draw.text(((i%3)*640+9,(i//3)*385+4),f'{i+1:02d} / 12',fill='#17384c')
        text=page.get_textpage().get_text_range();assert len(text)>35
        pages.append(dict(page=i+1,width_pt=w,height_pt=h,raster=str(path),raster_sha256=digest(path),pdf_text_characters=len(text)))
    contact.save(HERE/'contact_sheet.png')
    return dict(expected_pages=12,actual_pages=12,aspect_ratio='16:9',all_pages_rasterized=True,
                text_boxes_outside_page=outliers,overflow_check='WeasyPrint all text box bounds plus fixed12page gate; charts separately retain full data and axes; visual review still required.',pages=pages,
                contact_sheet=str(HERE/'contact_sheet.png'),contact_sheet_sha256=digest(HERE/'contact_sheet.png'))


def main():
    started=time.perf_counter();cpu=time.process_time()
    w,c=time.perf_counter(),time.process_time();values=load_sources();stage('source_copy_and_identity_scope',w,c)
    w,c=time.perf_counter(),time.process_time();write_slides(values);stage('all_point_charts_and_editable_HTML_MD',w,c)
    w,c=time.perf_counter(),time.process_time();qa=render_and_inspect();stage('PDF_render_all_page_raster_and_geometry_QA',w,c)
    receipt=dict(schema_version=1,status='rendered_working_draft_pending_ROOT_visual_review',as_of_utc='2026-11-06T17:30:00Z',
                 built_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),sources=source_evidence,
                 explicit_read_scope='Opened authorized primary/A/B/C410 CSVs and existing discussion/source metadata only. No C1B/C2.8B or behavior scores, GPU, CP or native model payloads.',
                 plots=plot_evidence,primary_equal_document_results_preserved=True,all_seed_lambda_draw_inputs_retained=True,
                 D_B4_equivalent_states_not_independent_replicates=True,QA=qa,nonoverlapping_stages=stages,
                 internal_child_build_wall_seconds=time.perf_counter()-started,internal_process_CPU_seconds=time.process_time()-cpu,
                 timing_semantics='Internal build child inclusive; stages nested. External whole Popen/resource timer is authoritative parent; authoring and monitoring wait not measured here.',
                 scientific_core_live_exports_current_report_or_existing_figures_modified=False)
    (HERE/'build_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    outputs=[]
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name!='output_manifest.json':outputs.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=digest(p)))
    (HERE/'output_manifest.json').write_text(json.dumps(dict(status='complete_build_outputs_not_final_thesis',files=outputs),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],pages=12,PDF=str(HERE/'答辩演示工作稿.pdf'),contact_sheet=str(HERE/'contact_sheet.png'),receipt_sha256=digest(HERE/'build_receipt.json')),ensure_ascii=False))


if __name__=='__main__':main()
