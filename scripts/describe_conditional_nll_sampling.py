#!/usr/bin/env python3
"""Analytic finite-pool NLL reference; no draw, bank, model, or test is run."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from replay_study.selection import _proportional_allocation


def main():
    project=Path(__file__).resolve().parents[1]
    statistics_path=project/"artifacts/generated/pool_seed11_stats.pt"
    source_bank=project/"artifacts/banks/generated_random_seed11.json"
    bank=json.loads(source_bank.read_text())
    statistics_sha=hashlib.sha256(statistics_path.read_bytes()).hexdigest()
    if statistics_sha!=bank["teacher_statistics_sha256"]:
        raise ValueError("statistics contents differ from the frozen parent bank")
    stats=torch.load(statistics_path,map_location="cpu",weights_only=False)
    values=np.asarray(stats["teacher_nll"],dtype=np.float64)
    counts=np.asarray(stats["prediction_token_counts"])
    if values.ndim!=1 or not np.isfinite(values).all() or not np.all(counts==counts[0]):
        raise ValueError("finite aligned equal-length chunk NLL required")
    if bank["pool_unique_chunks"]!=len(values) or bank["pool_chunks"]!=len(values):
        raise ValueError("finite population must contain exact-unique source chunks")
    strata=np.array_split(np.argsort(values,kind="stable"),5)
    sizes=np.asarray([len(indices) for indices in strata],dtype=np.int64)
    means=np.asarray([values[indices].mean() for indices in strata])
    variances=np.asarray([values[indices].var(ddof=1) for indices in strata])
    n=len(values);population_mean=float(values.mean());population_variance=float(values.var(ddof=1))
    records=[]
    for k in (20,196):
        allocation=_proportional_allocation(sizes,k)
        weights=allocation/k
        expected_mean=float(weights@means)
        variance=float(np.sum(weights**2*(1-allocation/sizes)*variances/allocation))
        records.extend([
            {"method":"random","bank_chunks":k,"expected_bank_nll_mean":population_mean,
             "conditional_bias_against_pool_mean":0.,
             "analytic_variance_bank_nll_mean":(1-k/n)*population_variance/k,
             "analytic_sd_bank_nll_mean":float(np.sqrt((1-k/n)*population_variance/k))},
            {"method":"nll_stratified","bank_chunks":k,"stratum_allocation":allocation.tolist(),
             "actual_stratum_weights":weights.tolist(),"expected_bank_nll_mean":expected_mean,
             "conditional_bias_against_pool_mean":expected_mean-population_mean,
             "analytic_variance_bank_nll_mean":variance,"analytic_sd_bank_nll_mean":float(np.sqrt(variance))}])
    output=project/"results/bank_diagnostics/conditional_nll_sampling_reference.json"
    payload={"scope":"Exact finite-population reference for chunk mean teacher NLL, conditional on frozen pool11",
             "not_claimed":"No prediction about gradient or trained-endpoint variance; no trained repeats added",
             "banks_generated":0,"model_or_test_inference_performed":False,
             "statistics_sha256":statistics_sha,"source_bank_manifest_sha256":hashlib.sha256(source_bank.read_bytes()).hexdigest(),
             "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "selection_source_sha256":hashlib.sha256((project/"src/replay_study/selection.py").read_bytes()).hexdigest(),
             "pool_chunks":n,"prediction_tokens_per_chunk":int(counts[0]),"pool_teacher_nll_mean":population_mean,
             "pool_teacher_nll_variance_ddof1":population_variance,"stratum_sizes":sizes.tolist(),
             "stratum_means":means.tolist(),"stratum_variances_ddof1":variances.tolist(),"records":records}
    output.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),"records":records},indent=2))


if __name__=="__main__":main()
