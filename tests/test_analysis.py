"""Hand-computed scientific invariants for analysis, independent of model code."""
import json
import math

import pytest

from replay_study.analysis import (aggregate_endpoints, collect_analysis, endpoint_metrics, mean_std,
                                   paired_document_bootstrap, paired_seed_differences,
                                   shared_preparation_costs, training_costs, write_analysis,
                                   matched_validation_loss,verify_recorded_execution,training_validation_execution)


def corpus(documents, kl=0.0):
    chunks = []
    for doc_id, contributions in documents.items():
        for count, nll in contributions:
            chunks.append({"doc_id": doc_id, "chunk_index": len(chunks),
                           "valid_prediction_tokens": count, "sum_nll": count*nll,
                           "sum_kl": count*kl})
    total = sum(r["valid_prediction_tokens"] for r in chunks)
    value = sum(r["sum_nll"] for r in chunks)
    return {"nll": value/total, "sum_nll": value, "prediction_tokens": total,
            "complete_corpus": True, "kl": kl, "chunks": chunks}


def three(value):
    return {name: corpus({"a": [(2, value)]}) for name in ("target", "c4", "wikitext")}


def test_gain_forgetting_signs_and_equal_document_estimand():
    initial = {name: corpus({"short": [(1, 2)], "long": [(9, 2)]}) for name in ("target", "c4", "wikitext")}
    current = {name: corpus({"short": [(1, 1)], "long": [(9, 3)]}, kl=.2) for name in initial}
    metrics = endpoint_metrics(current, initial)
    # Token-weighted endpoint NLL = (1 + 27)/10 = 2.8; equal-doc mean = 2.
    assert metrics["g_new"] == pytest.approx(-.8)
    assert metrics["f_old_c4"] == pytest.approx(.8)
    assert metrics["g_new_equal_document"] == pytest.approx(0)
    assert metrics["f_old_wikitext_equal_document"] == pytest.approx(0)
    assert metrics["kl_c4"] == pytest.approx(.2)


def test_sample_sd_is_not_sem_and_single_seed_is_undefined():
    stats = mean_std([1, 3, 5])
    assert stats == {"mean": 3, "std_ddof1": 2, "n": 3}
    assert mean_std([3])["std_ddof1"] is None
    assert mean_std([]) == {"mean": None, "std_ddof1": None, "n": 0}


def test_paired_differences_use_matching_seeds_before_sd():
    registry = [{"group": g, "seed": s, "ratio": .05, "weight": .5} for g in ("M", "B4") for s in (11,22,33)]
    rows = []
    # Large between-seed variation cancels: paired differences are exactly [1,2,3].
    for seed, base, effect in [(11,100,1), (22,500,2), (33,900,3)]:
        for group, value in [("B4",base),("M",base+effect)]:
            rows.append({"run_id": f"{group}{seed}", "group": group, "seed": seed, "ratio": .05,
                         "weight": .5, "g_new": value, "f_old_c4": -value})
    differences, aggregates = paired_seed_differences(rows, registry)
    gain = next(r for r in aggregates if r["metric"] == "g_new")
    assert [r["g_new"] for r in differences] == [1,2,3]
    assert gain["mean"] == 2 and gain["std_ddof1"] == 1 and gain["n"] == 3
    assert next(r for r in aggregates if r["metric"]=="f_old_c4")["mean"] == -2
    # Missing a partner cannot silently become an unpaired comparison.
    d, a = paired_seed_differences([r for r in rows if r["run_id"]!="B422"], registry)
    assert [r["seed"] for r in d] == [11,33]
    assert next(r for r in a if r["metric"]=="g_new")["complete"] is False


def test_document_bootstrap_weights_and_all_chunks_stay_together():
    a = corpus({"short": [(1,3)], "long": [(4,1),(5,1)]})
    b = corpus({"short": [(1,2)], "long": [(4,2),(5,2)]})
    result = paired_document_bootstrap(a,b,n_bootstrap=2000,seed=17)
    # Per-doc deltas are +1 and -1; token-weighted is -0.8, equal-doc is 0.
    assert result["token_weighted"]["estimate"] == pytest.approx(-.8)
    assert result["equal_document"]["estimate"] == pytest.approx(0)
    assert result["documents"] == 2  # Three chunks must not create three units.
    assert result["token_weighted"]["ci_low"] == -1
    assert result["token_weighted"]["ci_high"] == 1
    assert "does not replace" in result["scope"]
    # An additive constant between two highly heterogeneous checkpoints has
    # an exactly degenerate paired CI; independent model resampling would fail.
    c = corpus({"a": [(2,100)], "b": [(5,1000)]})
    d = corpus({"a": [(2,99.75)], "b": [(5,999.75)]})
    exact = paired_document_bootstrap(c,d,n_bootstrap=100,seed=3)
    assert exact["token_weighted"] == {"estimate": .25,"ci_low": .25,"ci_high": .25}
    assert exact["equal_document"] == {"estimate": .25,"ci_low": .25,"ci_high": .25}


def test_same_objective_source_pairs_match_seed_b_and_lambda():
    registry = [{"group": group, "seed": seed, "ratio": .05, "weight": .1}
                for group in ("B1", "B2", "B3", "B4") for seed in (11,22,33)]
    values = {"B1": 0., "B2": 2., "B3": 3., "B4": 7.}
    rows = [{**row, "run_id": f"{row['group']}{row['seed']}",
             "g_new": 100*row["seed"] + values[row["group"]]} for row in registry]
    pairs, stats = paired_seed_differences(rows, registry)
    for comparison, expected in (("B3-B1",3.),("B4-B2",5.)):
        selected = [r for r in pairs if r["comparison"] == comparison]
        assert len(selected) == 3 and [r["g_new"] for r in selected] == [expected]*3
        aggregate = next(r for r in stats if r["comparison"]==comparison and r["metric"]=="g_new")
        assert aggregate["n"]==3 and aggregate["mean"]==expected and aggregate["std_ddof1"]==0
    changed = [dict(r) for r in rows]
    next(r for r in changed if r["group"]=="B3" and r["seed"]==22)["weight"] = .5
    with pytest.raises(ValueError,match="outside"):
        paired_seed_differences(changed,registry)


def test_bootstrap_rejects_misalignment_and_chunk_without_document():
    a = corpus({"a": [(2,1)]}); b = corpus({"b": [(2,1)]})
    with pytest.raises(ValueError,match="document sets"):
        paired_document_bootstrap(a,b,n_bootstrap=10)
    b = corpus({"a": [(3,1)]})
    with pytest.raises(ValueError,match="chunk identities"):
        paired_document_bootstrap(a,b,n_bootstrap=10)
    b = corpus({"a": [(2,1)]});del b["chunks"][0]["doc_id"]
    with pytest.raises(ValueError,match="doc_id"):
        paired_document_bootstrap(a,b,n_bootstrap=10)


def test_costs_do_not_double_count_teacher_or_cluster_and_exclude_pilot():
    summary = {"counters": {"setup_seconds": 1,"training_seconds": 10,"validation_seconds": 4,
                            "checkpoint_seconds": 2,"total_accounted_wall_seconds": 18,
                            "training_teacher_forward_seconds": 3,"validation_teacher_forward_seconds": 1,
                            "validation_kl_arithmetic_seconds": 1}}
    costs = training_costs(summary)
    assert costs["parent_component_sum_seconds"] == 17
    assert costs["unclassified_wall_seconds"] == 1
    assert costs["training_teacher_forward_seconds_nested"] == 3
    prep = {"pools": {"11": {"role":"formal","generation":{"generation_wall_seconds":10},
                              "teacher_statistics":{"wall_seconds":2},
                              "banks":{"cluster":{"wall_seconds":3,"clustering_wall_seconds":2.5}}},
                      "101":{"role":"pilot"}}}
    rows = shared_preparation_costs(prep)
    assert sum(r["wall_seconds"] for r in rows if r["additive"]) == 15
    assert len({r["seed"] for r in rows}) == 1


def write_fixture(tmp_path, completed=True):
    baseline = tmp_path/"baseline.json";baseline.write_text(json.dumps({"revision":"pinned","corpora":three(2)}))
    path = tmp_path/"formal_summary.json"
    config = {"seed":11,"replay_ratio":.05,"replay_weight":.5,"learning_rate":3e-5,
              "target_prediction_tokens":100,"model_revision":"pinned","run_label":"formal_B4"}
    summary = {"status":"completed" if completed else "interrupted","config":config,
               "checkpoint_path":str(tmp_path/"latest.pt"),
               "counters":{"step":2,"target_prediction_tokens":100 if completed else 50,"replay_prediction_tokens":5},
               "validation_trajectory":[{"step":2,"target_prediction_tokens":100 if completed else 50,
                                         "replay_prediction_tokens":5,"metrics":three(1.5)}]}
    path.write_text(json.dumps(summary))
    registry = tmp_path/"registry.json"
    registry.write_text(json.dumps([{"id":"formal_B4","phase":"formal","group":"B4","seed":11,
                                     "ratio":.05,"weight":.5,"learning_rate":3e-5,"summary_path":str(path)}]))
    return registry,baseline,path


def test_collection_never_promotes_incomplete_run_and_refuses_final_output(tmp_path):
    registry, baseline, _ = write_fixture(tmp_path,completed=False)
    bundle = collect_analysis(registry,baseline,analysis_status="active")
    assert bundle["endpoints"] == [] and len(bundle["trajectories"]) == 1
    assert len(bundle["observed_run_costs"])==1 and bundle["training_costs"]==[]
    assert bundle["observed_run_costs"][0]["target_prediction_tokens"]==50
    assert bundle["metadata"]["complete"] is False
    with pytest.raises(ValueError,match="incomplete"):
        write_analysis(bundle,tmp_path/"out",plots=False)
    write_analysis(bundle,tmp_path/"out",allow_partial=True,plots=False)
    assert read_metadata(tmp_path/"out")["completed_endpoints"] == 0


def read_metadata(path):
    return json.loads((path/"analysis_manifest.json").read_text())


def test_collect_formal_endpoint_and_reject_pilot_or_wrong_revision(tmp_path):
    registry, baseline, summary_path = write_fixture(tmp_path)
    bundle = collect_analysis(registry,baseline,analysis_status="active")
    assert bundle["endpoints"][0]["g_new"] == .5
    assert bundle["metadata"]["complete"] is True
    registry_data=json.loads(registry.read_text());registry_data[0]["phase"]="pilot";registry.write_text(json.dumps(registry_data))
    with pytest.raises(ValueError,match="formal runs only"):
        collect_analysis(registry,baseline,analysis_status="active")
    registry_data[0]["phase"]="formal";registry.write_text(json.dumps(registry_data))
    summary=json.loads(summary_path.read_text());summary["config"]["model_revision"]="wrong";summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="revision mismatch"):
        collect_analysis(registry,baseline,analysis_status="active")


def test_future_evaluation_interface_is_explicit_and_preserves_validation_trajectory(tmp_path):
    registry, baseline, _ = write_fixture(tmp_path)
    root=tmp_path/"evaluations";(root/"formal_B4").mkdir(parents=True)
    payload={"revision":"pinned","checkpoint":str(tmp_path/"latest.pt"),"corpora":three(1.25)}
    (root/"formal_B4"/"final.json").write_text(json.dumps(payload))
    (root/"initial_pythia-410m.json").write_text(baseline.read_text())
    bundle=collect_analysis(registry,baseline,evaluation_root=root,n_bootstrap=20,analysis_status="active")
    assert bundle["endpoints"][0]["g_new"] == .75
    assert bundle["trajectories"][0]["g_new"] == .5
    assert bundle["metadata"]["split"] == "test"
    target = [r for r in bundle["document_bootstrap_endpoints"] if r["metric"]=="g_new"]
    assert len(target)==2 and all(r["estimate"]==.75 for r in target)
    assert all(r["ci_low"]==.75 and r["ci_high"]==.75 for r in target)


def test_provisional_status_cannot_become_a_final_scientific_report(tmp_path):
    registry,baseline,_=write_fixture(tmp_path)
    bundle=collect_analysis(registry,baseline,analysis_status="provisional-numerical-noise")
    assert bundle["metadata"]["complete"] is False
    assert bundle["metadata"]["scientific_comparisons_suppressed"] is True
    assert bundle["paired_statistics"] == []
    with pytest.raises(ValueError,match="incomplete"):
        write_analysis(bundle,tmp_path/"final",plots=False)
    with pytest.raises(ValueError,match="active, frozen"):
        collect_analysis(registry,baseline,analysis_status="provisional-numerical-noise",evaluation_root=tmp_path)


def test_replay_budget_and_selected_checkpoint_are_verified(tmp_path):
    registry,baseline,path=write_fixture(tmp_path)
    summary=json.loads(path.read_text());summary["counters"]["replay_prediction_tokens"]=4;path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="replay exposure"):
        collect_analysis(registry,baseline,analysis_status="active")
    summary["counters"]["replay_prediction_tokens"]=5;path.write_text(json.dumps(summary))
    root=tmp_path/"eval";(root/"formal_B4").mkdir(parents=True)
    (root/"initial_pythia-410m.json").write_text(baseline.read_text())
    (root/"formal_B4"/"final.json").write_text(json.dumps({"revision":"pinned","checkpoint":"/wrong/latest.pt","corpora":three(1)}))
    with pytest.raises(ValueError,match="different checkpoint"):
        collect_analysis(registry,baseline,evaluation_root=root,n_bootstrap=10,analysis_status="active")


def test_active_registry_cannot_reuse_old_nondeterministic_config_or_wrong_run(tmp_path):
    registry,baseline,path=write_fixture(tmp_path)
    rows=json.loads(registry.read_text())
    frozen=json.loads(path.read_text())["config"];frozen["deterministic_algorithms"]=True
    config_path=tmp_path/"active_config.json";config_path.write_text(json.dumps(frozen))
    rows[0]["config_path"]=str(config_path);registry.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="deterministic_algorithms"):
        collect_analysis(registry,baseline,analysis_status="active")
    del rows[0]["config_path"];rows[0]["id"]="deterministic_B4";registry.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="run_label mismatch"):
        collect_analysis(registry,baseline,analysis_status="active")
    rows[0]["id"]="formal_B4";rows[0]["status"]="provisional";registry.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="provisional"):
        collect_analysis(registry,baseline,analysis_status="active")


def test_best_checkpoint_exposure_is_early_but_full_training_cost_is_retained(tmp_path):
    registry,baseline,path=write_fixture(tmp_path)
    summary=json.loads(path.read_text());best_path=str(tmp_path/"checkpoint_best.pt")
    summary["validation_trajectory"].insert(0,{"step":1,"target_prediction_tokens":50,
                                             "replay_prediction_tokens":2,"metrics":three(1.4)})
    summary["best_target_validation"]={"step":1,"target_prediction_tokens":50,"nll":1.4,"checkpoint_path":best_path}
    path.write_text(json.dumps(summary))
    root=tmp_path/"eval";(root/"formal_B4").mkdir(parents=True)
    (root/"initial_pythia-410m.json").write_text(baseline.read_text())
    (root/"formal_B4"/"best.json").write_text(json.dumps({"revision":"pinned","checkpoint":best_path,"corpora":three(1.4)}))
    bundle=collect_analysis(registry,baseline,evaluation_root=root,checkpoint="best",n_bootstrap=10,analysis_status="active")
    endpoint=bundle["endpoints"][0]
    assert endpoint["checkpoint_step"]==1 and endpoint["target_prediction_tokens"]==50
    assert endpoint["replay_prediction_tokens"]==2
    assert endpoint["full_training_target_prediction_tokens"]==100
    assert endpoint["full_training_replay_prediction_tokens"]==5
    assert bundle["training_costs"][0]["target_prediction_tokens"]==100
    assert "different exposure" in bundle["metadata"]["checkpoint_exposure_policy"]


def test_endpoint_rejects_different_chunks_even_if_document_token_totals_match():
    initial=three(2);current=three(1)
    current["target"]["chunks"][0]["chunk_index"]=99
    with pytest.raises(ValueError,match="chunk identities"):
        endpoint_metrics(current,initial)


def test_matched_loss_uses_first_downward_crossing_and_marks_nonmonotonicity():
    def point(exposure,nll,forgetting):
        return {"step":exposure//10,"target_prediction_tokens":exposure,"nll_target":nll,
                "f_old_c4":forgetting,"f_old_wikitext":forgetting}
    a=[point(10,2.5,.1),point(20,1.5,.3),point(30,2.5,.8),point(40,1.,.9)]
    b=[point(10,2.5,.2),point(20,.5,.4)]
    result=matched_validation_loss(a,b,initial_target_nll=3.,n_levels=1)
    assert result["common_loss_low"]==1. and result["common_loss_high"]==3.
    assert result["nonmonotonic_a"] is True and result["nonmonotonic_b"] is False
    p=result["points"][0]
    assert p["target_validation_nll"]==2.
    assert p["interpolated_target_exposure_a"]==15.
    assert p["interpolated_target_exposure_b"]==12.5
    assert p["f_old_c4_a"]==pytest.approx(.2) and p["f_old_c4_b"]==pytest.approx(.25)
    assert p["delta_f_old_c4"]==pytest.approx(-.05)
    assert p["downward_crossings_a"]==2
    assert p["actual_checkpoint"] is False and p["equal_budget"] is False
    assert p["bracket_target_low_a"]==10 and p["bracket_target_high_a"]==20


def test_matched_loss_does_not_extrapolate_or_consume_test_values():
    worse=[{"target_prediction_tokens":10,"nll_target":3.1}]
    better=[{"target_prediction_tokens":10,"nll_target":2.}]
    result=matched_validation_loss(worse,better,initial_target_nll=3.)
    assert result["status"]=="no_common_improved_loss_range" and result["points"]==[]
    with pytest.raises(ValueError,match="validation trajectories only"):
        matched_validation_loss([{**better[0],"split":"test"}],better,initial_target_nll=3.)


def test_protocol_status_authority_and_live_event_identity_block_cross_batch_mix(tmp_path):
    registry,baseline,path=write_fixture(tmp_path)
    status=tmp_path/"protocol_status.json";status.write_text(json.dumps({"active_registry":None}))
    with pytest.raises(ValueError,match="no frozen active registry"):
        collect_analysis(registry,baseline,analysis_status="active",protocol_status_path=status)
    status.write_text(json.dumps({"active_registry":str(registry)}))
    assert collect_analysis(registry,baseline,analysis_status="active",protocol_status_path=status)["metadata"]["complete"]
    status.write_text(json.dumps({"active_registry":str(tmp_path/"other_registry.json")}))
    with pytest.raises(ValueError,match="not the active registry"):
        collect_analysis(registry,baseline,analysis_status="active",protocol_status_path=status)
    # A new ID pointing to an old live event folder must fail even before a summary exists.
    rows=json.loads(registry.read_text());run=tmp_path/"run";run.mkdir()
    summary=json.loads(path.read_text());path.unlink();rows[0]["output_dir"]=str(run);rows[0]["id"]="new_deterministic_B4"
    registry.write_text(json.dumps(rows))
    events=[{"event":"start","config":summary["config"]},
            {"event":"validation",**summary["validation_trajectory"][0]}]
    (run/"events.jsonl").write_text("\n".join(json.dumps(r) for r in events)+"\n")
    with pytest.raises(ValueError,match="run_label mismatch"):
        collect_analysis(registry,baseline,analysis_status="active")


def test_execution_policy_mismatch_and_missing_production_baseline_are_rejected(tmp_path):
    policy={"torch_version":"test","cuda_version":"test","precision":"bf16","attention_implementation":"sdpa",
            "effective_precision":"cuda_bfloat16_autocast","device_type":"cuda","gpu_name":"NVIDIA H100 80GB HBM3",
            "attention_backend_requested":"auto","flash_sdp_enabled":True,"mem_efficient_sdp_enabled":True,
            "math_sdp_enabled":True,"cudnn_sdp_enabled":True,"deterministic_algorithms":True,"deterministic_warn_only":False,
            "allow_tf32_matmul":True,"allow_tf32_cudnn":True,"cudnn_deterministic":True,"cudnn_benchmark":False,
            "CUBLAS_WORKSPACE_CONFIG":":4096:8","model_weight_dtype":"torch.float32"}
    verify_recorded_execution(policy,dict(policy))
    with pytest.raises(ValueError,match="allow_tf32_matmul"):
        verify_recorded_execution(policy,{**policy,"allow_tf32_matmul":False})
    with pytest.raises(ValueError,match="model_weight_dtype"):
        verify_recorded_execution(policy,{**policy,"model_weight_dtype":"torch.bfloat16"})
    with pytest.raises(ValueError,match="cudnn_sdp_enabled"):
        verify_recorded_execution(policy,{**policy,"cudnn_sdp_enabled":False})
    registry,baseline,_=write_fixture(tmp_path)
    with pytest.raises(ValueError,match="requires execution_provenance"):
        collect_analysis(registry,baseline,analysis_status="active",require_execution_provenance=True)


def test_saved_device_and_effective_precision_mapping_is_explicit():
    summary={"config":{"precision":"bf16","device":"cuda:0"},"provenance":{
        "precision":"bf16","device":"cpu","gpu":None,"model_weight_dtype":"torch.float32",
        "allow_tf32":True,"allow_tf32_matmul":False,"allow_tf32_cudnn":True},"validation_trajectory":[]}
    mapped=training_validation_execution(summary)
    assert mapped["device_type"]=="cpu" and mapped["effective_precision"]=="float32" and mapped["gpu_name"] is None
    assert mapped["allow_tf32_matmul"] is False and mapped["allow_tf32_cudnn"] is True
    summary["provenance"].update(device="cuda:2",gpu="NVIDIA H100 80GB HBM3")
    mapped=training_validation_execution(summary)
    assert mapped["effective_precision"]=="cuda_bfloat16_autocast" and mapped["device_type"]=="cuda"
    assert mapped["gpu_name"]=="NVIDIA H100 80GB HBM3"
