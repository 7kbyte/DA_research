"""Conditional draw identities and provenance guards; no model or GPU work."""
import json
import math
from pathlib import Path

import pytest
import torch

from replay_study.analysis import collect_analysis,sha256
from replay_study.bank_variance import (PHASE,analyze_bank_variance,completed_parent,make_bank_variance,
                                        _conditional_pairs,_conditional_statistics)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def parent_fixture(tmp_path):
    pool=tmp_path/"pool.pt";stats=tmp_path/"stats.pt";parent_bank=tmp_path/"parent_bank.pt"
    generator=torch.Generator().manual_seed(2)
    torch.save({"input_ids":torch.randint(0,250,(12,512),generator=generator,dtype=torch.int32)},pool)
    write(pool.with_suffix(".json"),{"config":{"seed":11},"experiment_role":"formal","revision":"pinned","split":"generated_training_candidates","artifact_sha256":sha256(pool)})
    torch.save({"teacher_nll":torch.linspace(1,3,12),"features":torch.randn(12,6,generator=generator),
                "prediction_token_counts":torch.full((12,),511),"source_pool_sha256":sha256(pool),"revision":"pinned"},stats)
    write(parent_bank.with_suffix(".json"),{"source_pool":str(pool),"source_pool_sha256":sha256(pool),
          "teacher_statistics_artifact":str(stats),"teacher_statistics_sha256":sha256(stats)})
    config={"model_name_or_path":"unused-pinned-model","model_revision":"pinned","target_path":"unused-target.pt",
            "seed":11,"replay_ratio":.05,"replay_weight":.5,"replay_objective":"kl","replay_path":str(parent_bank),
            "learning_rate":3e-5,"target_prediction_tokens":10_000_000,"deterministic_algorithms":True,
            "precision":"bf16","attention_implementation":"sdpa","attention_backend":"auto","allow_tf32":True,
            "output_dir":str(tmp_path/"parent_run"),"run_label":"parent_B4"}
    config_path=write(tmp_path/"parent_config.json",config)
    summary={"status":"completed","config":config,"counters":{"step":306,"target_prediction_tokens":10_000_000,
              "replay_prediction_tokens":500_000,"total_accounted_wall_seconds":500},
              "provenance":{"deterministic_algorithms":True,"deterministic_warn_only":False}}
    summary_path=write(tmp_path/"parent_summary.json",summary)
    registry=write(tmp_path/"parent_registry.json",[{"id":"parent_B4","group":"B4","seed":11,"ratio":.05,"weight":.5,
                   "learning_rate":3e-5,"phase":"formal","active":True,"config_path":str(config_path),"summary_path":str(summary_path)}])
    status=write(tmp_path/"protocol_status.json",{"active_registry":str(registry)})
    return registry,status,summary_path


def make_fixture(tmp_path):
    registry,status,_=parent_fixture(tmp_path)
    output=tmp_path/"conditional"
    manifest=make_bank_variance(registry,status,output,run_root=tmp_path/"runs",result_root=tmp_path/"results",
                                bank_sizes=(2,3),selection_seeds=(1001,1002))
    return output,manifest


def test_provisional_and_partial_parent_cannot_prepare_banks(tmp_path):
    registry,status,summary_path=parent_fixture(tmp_path)
    rows=json.loads(registry.read_text());rows[0]["active"]=False;registry.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="active formal"):
        completed_parent(registry,status)
    rows[0]["active"]=True;registry.write_text(json.dumps(rows))
    summary=json.loads(summary_path.read_text());summary["status"]="interrupted";summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="incomplete"):
        make_bank_variance(registry,status,tmp_path/"must_not_exist",run_root=tmp_path/"r",result_root=tmp_path/"s")
    assert not (tmp_path/"must_not_exist").exists()


def test_prepared_matrix_preserves_train_seed_and_never_creates_runnable_queue(tmp_path):
    output,manifest=make_fixture(tmp_path)
    rows=json.loads((output/"registry.json").read_text())
    assert len(rows)==12 and len(manifest["banks"])==12
    assert {r["training_seed"] for r in rows}=={11} and {r["generation_seed"] for r in rows}=={11}
    assert {r["selection_seed"] for r in rows}=={1001,1002}
    assert all(r["phase"]==PHASE and "seed" not in r and r["active"] is False for r in rows)
    configs=[json.loads(Path(r["config_path"]).read_text()) for r in rows]
    assert {c["seed"] for c in configs}=={11} and {c["target_prediction_tokens"] for c in configs}=={10_000_000}
    assert {c["replay_weight"] for c in configs}=={.5} and {c["learning_rate"] for c in configs}=={3e-5}
    assert manifest["training_started"] is False and manifest["model_or_test_inference_performed"] is False
    queue=json.loads((output/"queue_template.json").read_text())
    assert isinstance(queue,dict) and queue["launch_authorized"] is False and len(queue["jobs"])==12
    assert all("--summary-copy" in job["argv"] for job in queue["jobs"])
    for row in rows:
        bank=json.loads(Path(row["bank_path"]).with_suffix(".json").read_text())
        assert bank["bank_unique_chunks"]==row["bank_chunks"]
        assert bank["unique_prediction_tokens"]==row["bank_chunks"]*511
        assert sum(bank["nll_composition"]["pool_nll_stratum_counts"])==row["bank_chunks"]
    with pytest.raises(ValueError,match="formal runs only"):
        collect_analysis(output/"registry.json",tmp_path/"unused_baseline",analysis_status="active")


def test_conditional_pairs_are_draw_differences_not_training_seed_repeats():
    rows=[]
    for draw,base,effect in [(1001,100,1),(1002,500,2)]:
        for group,value in [("B4",base),("M",base+effect)]:
            rows.append({"group":group,"bank_chunks":20,"selection_seed":draw,"teacher_nll_mean":value})
    pairs=_conditional_pairs(rows,["teacher_nll_mean"])
    stats=_conditional_statistics(pairs,["teacher_nll_mean"],2)[0]
    assert stats["mean"]==1.5 and stats["std_ddof1"]==pytest.approx(math.sqrt(.5))
    assert stats["n"]==2 and stats["n_training_seeds"]==1 and stats["n_generation_pools"]==1
    assert stats["replicate_unit"]=="selection_seed"


def test_statistics_content_is_bound_to_completed_parent_before_output_creation(tmp_path):
    registry,status,_=parent_fixture(tmp_path)
    statistics_path=tmp_path/"stats.pt"
    statistics=torch.load(statistics_path,weights_only=False)
    statistics["teacher_nll"]=statistics["teacher_nll"]+1
    # Matching source-pool SHA/revision do not permit changing frozen teacher scores.
    torch.save(statistics,statistics_path)
    output=tmp_path/"must_not_exist"
    with pytest.raises(ValueError,match="frozen statistics"):
        make_bank_variance(registry,status,output,run_root=tmp_path/"runs",result_root=tmp_path/"results")
    assert not output.exists()


def test_prepared_only_analysis_is_explicit_partial_and_scientific_contents_are_bound(tmp_path):
    output,_=make_fixture(tmp_path)
    baseline=write(tmp_path/"baseline.json",{"corpora":{}})  # No trained endpoints: no score is needed or invented.
    with pytest.raises(ValueError,match="incomplete"):
        analyze_bank_variance(output/"registry.json",baseline,tmp_path/"analysis",plots=False)
    result=analyze_bank_variance(output/"registry.json",baseline,tmp_path/"analysis",allow_partial=True,plots=False)
    assert result["completed_draw_runs"]==0 and result["expected_draw_runs"]==12
    assert result["training_seeds"]==[11] and result["test_scores_accessed"] is False
    rows=json.loads((output/"registry.json").read_text());rows[0]["active"]=True
    (output/"registry.json").write_text(json.dumps(rows))
    # Administrative activation is allowed, while changing the scientific draw is not.
    analyze_bank_variance(output/"registry.json",baseline,tmp_path/"analysis",allow_partial=True,plots=False)
    rows[0]["selection_seed"]=9999;(output/"registry.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="scientific contents changed"):
        analyze_bank_variance(output/"registry.json",baseline,tmp_path/"analysis",allow_partial=True,plots=False)
