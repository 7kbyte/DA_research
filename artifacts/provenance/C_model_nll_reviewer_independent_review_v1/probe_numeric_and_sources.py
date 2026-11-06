#!/usr/bin/env python3
"""Independent asymmetric toy mathematics and source checks; no actual scores."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
path=ROOT/'scripts/review_C_model_nll_results_v2.py'
raw=path.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='7a48fb77a07d5d0d757c7f67f473dd49d4e262ffe497dfbe655184211be3d727'
spec=importlib.util.spec_from_file_location('independent_C_model_nll_reviewer',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old=ROOT/'artifacts/provenance/C_model_nll_review_v2_schema_repair_v1/retained/scripts/review_C_model_nll_results_v2.py'
prior=old.read_bytes()
assert hashlib.sha256(prior).hexdigest()=='96665a639bccff9420425e3253b9483fb99941722e247d87a1f4a7157e6f4e5b'
assert prior.replace(b"authority.get('authorized_by')=='ROOT'",b"authority.get('authorized_by')=='root'")==raw
before=ast.parse((ROOT/'scripts/review_C_nll_results.py').read_bytes());after=ast.parse(raw)
nodes=('stats','favorable','documents','delta_vectors','raw_metrics','Resampling','validate_grid')
for name in nodes:
    a=next(n for n in before.body if getattr(n,'name',None)==name)
    b=next(n for n in after.body if getattr(n,'name',None)==name)
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)

# Three unequal-length documents; asymmetric nonconstant NLL differences.
# Build manual index draws independently of multiplicity-matrix implementation.
names=['a','b','c'];counts=np.asarray([1.,2.,7.]);delta=np.asarray([2.,-2.,21.])
indices=np.random.default_rng(20261106).integers(0,3,size=(5000,3))
token_draws=np.asarray([sum(delta[i] for i in row)/sum(counts[i] for i in row) for row in indices])
equal_draws=np.asarray([statistics.mean(delta[i]/counts[i] for i in row) for row in indices])
ctx=m.Resampling(names,counts)
checks=[]
for sign in (1,-1):
    actual=ctx.interval(names,counts,delta,sign)
    for weighting,draws,point in [('token_weighted',token_draws,21/10),('equal_document',equal_draws,4/3)]:
        lo,hi=np.quantile(sign*draws,[.025,.975])
        expected=dict(estimate=sign*point,ci_low=float(lo),ci_high=float(hi))
        assert all(abs(actual[weighting][k]-v)<1e-12 for k,v in expected.items())
        checks.append(dict(sign=sign,weighting=weighting,actual=actual[weighting],manual_expected=expected))
positive=ctx.interval(names,counts,delta,1);negative=ctx.interval(names,counts,delta,-1)
assert all(negative[k]['ci_low']==-positive[k]['ci_high'] and negative[k]['ci_high']==-positive[k]['ci_low'] for k in positive)
assert m.stats([1,4,7])==dict(mean=4,std_ddof1=3.,n=3)
for model,logical,pairs in [('pythia-1b',48,6),('pythia-2.8b',30,3)]:
    p=m.PROFILES[model]
    grid=[dict(checkpoint=k,group=g,training_seed=s) for k in ('final','best') for g in p['groups'] for s in m.SEEDS]
    m.validate_grid(grid,p['groups']);assert len(grid)==logical and len(p['pairs'])==pairs
    bad=grid[:-1]
    try:m.validate_grid(bad,p['groups'])
    except ValueError:pass
    else:raise AssertionError('Partial model grid accepted')

# Read only already generated SYNTHETIC plots for independent visual inspection.
base=ROOT/'artifacts/provenance/C_model_nll_review_v2_preparation/toy_cli_stable_source/toy_cli_inputs'
image_sources=[];tiles=[]
for model,mode in [('pythia-1b','six'),('pythia-2.8b','none')]:
    for name in ('all_endpoints_token_7in.png','all_endpoints_equal_document_7in.png',
                 'paired_token_7in.png','paired_equal_document_7in.png'):
        source=base/model/mode/'review'/name
        payload=source.read_bytes()
        image_sources.append(dict(path=str(source),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
        with Image.open(source) as im:
            tile=Image.new('RGB',(650,640),'white')
            picture=im.convert('RGB');picture.thumbnail((650,610));tile.paste(picture,((650-picture.width)//2,26))
            ImageDraw.Draw(tile).text((8,7),model+' '+name,fill='black')
            tiles.append(tile)
sheet=Image.new('RGB',(2600,1280),'white')
for i,tile in enumerate(tiles):sheet.paste(tile,((i%4)*650,(i//4)*640))
sheet.save(HERE/'synthetic_contact_8.png')
result=dict(status='passed_independent_asymmetric_math_and_sources_only',
    production_source_sha256=hashlib.sha256(raw).hexdigest(),
    production_delta_exact_single_authority_literal=True,seven_numerical_AST_nodes_unchanged=list(nodes),
    frozen_bootstrap_replicates=5000,frozen_seed=20261106,independent_manual_intervals=checks,
    negative_G_reverses_CI_bounds=True,sample_SD_manual_check=True,
    complete_1B48_2p8B30_grids_and_partial_refusal=True,
    all_eight_synthetic_plot_sources=image_sources,
    actual_score_header_or_checkpoint_payload_bytes_read=0,
    actual_reviewer_gate_or_numeric_run_called=False,original_check_or_analyze_called=False,
    original_or_frozen_sources_modified=False)
with (HERE/'probe_result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],manual_interval_cases=len(checks),synthetic_plots=len(image_sources))))
