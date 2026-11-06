#!/usr/bin/env python3
"""Reconstruct the authorized B target weighting explanation, without resampling."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import statistics
import time

SPEC=importlib.util.spec_from_file_location("B_review",Path(__file__).with_name("review_supplement_B_test.py"))
review=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(review)


def decomposition(baseline, by_condition):
    counts=Counter(n for _,n in baseline.values());n_docs=len(baseline);n_tokens=sum(n for _,n in baseline.values());out=[]
    for (k,g), draws in sorted(by_condition.items()):
        gains=defaultdict(list)
        for current in draws:
            if {d:n for d,(_,n) in current.items()}!={d:n for d,(_,n) in baseline.items()}:
                raise ValueError("Document identities/counts differ")
            for doc,(nll,n) in current.items():gains[n].append(baseline[doc][0]-nll)
        for n,values in sorted(gains.items()):
            if n%511:raise ValueError("This B explanatory table assumes complete 511-prediction-token chunks")
            gain=statistics.mean(values);docs=counts[n]
            out.append({"bank_chunks":k,"group":g,"evaluation_chunks_per_doc":n//511,"documents":docs,"prediction_tokens_per_doc":n,
                "document_weight_fraction":docs/n_docs,"token_weight_fraction":docs*n/n_tokens,
                "mean_G_across_matching_documents_and_10_draws":gain,"contribution_to_equal_document_G":gain*docs/n_docs,
                "contribution_to_token_G":gain*docs*n/n_tokens})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review-receipt',type=Path,required=True);p.add_argument('--review-sha256',required=True)
    p.add_argument('--table',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.perf_counter();cpu=time.process_time()
    if review.sha(a.review_receipt)!=a.review_sha256:raise ValueError("Review proof changed")
    receipt=review.read(a.review_receipt);complete=review.read(receipt['authorized_COMPLETE']['path']);plan=review.read(complete['plan']['path']);manifest=review.read(plan['manifest']['path'])
    assert receipt['review_complete'] and plan['contract_id']=='supplement_B_nll'
    endpoints=review.rows(a.review_receipt.parent/'all_60_endpoints.csv');review.validate_inventory(endpoints);identity={r['run_id']:r for r in endpoints}
    source_paths=[next(iter(manifest['baseline_groups'].values()))['initial_output']]+[r['output'] for r in manifest['records']]
    for path in source_paths:
        if review.sha(path)!=receipt['verified_collector_source_sha256'][path]:raise ValueError("Authorized score bytes changed")
    baseline=review.document_means(review.read(source_paths[0])['corpora']['target']);grouped=defaultdict(list)
    for r in manifest['records']:
        endpoint=identity[r['run_id']];grouped[int(endpoint['bank_chunks']),r['group']].append(review.document_means(review.read(r['output'])['corpora']['target']))
    computed=decomposition(baseline,grouped);existing=review.rows(a.table)
    assert len(computed)==len(existing)==24
    errors=[]
    for row,old in zip(computed,existing):
        for key,value in row.items():
            if isinstance(value,str):assert old[key]==value
            else:errors.append(abs(float(old[key])-value))
    for k,g in grouped:
        subset=[r for r in computed if r['bank_chunks']==k and r['group']==g];es=[r for r in endpoints if int(r['bank_chunks'])==k and r['group']==g]
        for metric,field in [('g_new','contribution_to_token_G'),('g_new_equal_document','contribution_to_equal_document_G')]:
            errors.append(abs(sum(r[field] for r in subset)-statistics.mean(float(r[metric]) for r in es)))
    assert max(errors)<1e-10
    review.write_json(a.output,{'verified':True,'review_receipt':{'path':str(a.review_receipt.resolve()),'sha256':a.review_sha256},
        'source':{'path':str(Path(__file__).resolve()),'sha256':review.sha(__file__)},'table':{'path':str(a.table.resolve()),'sha256':review.sha(a.table)},
        'row_count':len(computed),'scalar_checks':len(errors),'maximum_absolute_error':max(errors),'new_bootstrap_or_forward_calls':0,
        'wall_seconds':time.perf_counter()-start,'process_CPU_seconds':time.process_time()-cpu,
        'timing_scope':'Independent 61 score SHA checks and explanatory weighting reconstruction, not the earlier uninstrumented table construction or original collector'})
    print(json.dumps({'verified':True,'output':str(a.output),'maximum_absolute_error':max(errors)}))


if __name__=='__main__':main()
