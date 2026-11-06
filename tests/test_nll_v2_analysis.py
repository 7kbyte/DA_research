"""Synthetic sealed-score fixtures; never open actual study evaluation outputs."""
import copy
import json
from pathlib import Path

import pytest

from replay_study.nll_v2_analysis import (EXECUTION_KEYS, MODEL_REVISIONS, collect_scores, prepare_plan, sha)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture(tmp_path, *, role="primary", alias=False, best=False):
    phase = {"primary": "formal", "supplement_A": "supplement_A_lambda_sweep", "supplement_B": "conditional_bank_variance"}[role]
    kind = "best" if best else "final"
    registry, records = [], []
    for index, group in enumerate(("B4", "M")):
        row = dict(id=group, phase=phase, active=True, group=group, ratio=.05, weight=.05, learning_rate=.0001, seed=11)
        if role == "supplement_B":
            row.pop("seed")
            row.update(training_seed=11, generation_seed=11, selection_seed=1001, bank_chunks=20)
        if alias:
            row.update(execution_kind="alias", alias_of="physical_" + group)
        registry.append(row)
        records.append(dict(id=group + "__" + kind, run_id=group, actual_run_id="physical_" + group if alias else group,
            phase=phase, phase_name="cohort", role=role, data_family="main_v3", model_tag="pythia-410m", kind=kind,
            group=group, ratio=.05, weight=.05, learning_rate=.0001, training_seed=11,
            selection_seed=1001 if role == "supplement_B" else None, generation_seed=11 if role == "supplement_B" else None,
            checkpoint="checkpoint_" + group, checkpoint_sha256="weights" if alias else group,
            weights_identity="checkpoint:weights" if alias else "checkpoint:" + group,
            source_model_equivalence=None, baseline_group="base", initial_job_id="initial", inference_job_id="shared" if alias else group,
            output=str(tmp_path / (group + ".json")), is_training_alias=alias,
            target_prediction_tokens_at_checkpoint=100 if best else 1000, replay_prediction_tokens_at_checkpoint=5 if best else 50,
            full_training_committed_path=dict(target_prediction_tokens=1000, replay_prediction_tokens=50,
                parent_wall_seconds=dict(training_seconds=2, validation_seconds=1)), step=1))
    spec_path, registry_path = tmp_path / "spec.json", tmp_path / "registry.json"
    write(spec_path, {}); write(registry_path, registry)
    policy = {k: False for k in EXECUTION_KEYS}
    policy.update(model="base-model", revision=MODEL_REVISIONS["pythia-410m"], precision="bf16", attention_backend="auto",
        torch_version="synthetic", cuda_version="synthetic", gpu_name="synthetic", device_type="cuda", effective_precision="cuda_bfloat16_autocast",
        attention_implementation="sdpa", model_weight_dtype="torch.float32", CUBLAS_WORKSPACE_CONFIG=":4096:8", deterministic_algorithms=True)
    group = dict(id="base", model_tag="pythia-410m", inference_policy=policy,
        test_paths={c: c + ".pt" for c in ("target", "c4", "wikitext")},
        evaluation_input_sha256={c + ".pt": c + "sha" for c in ("target", "c4", "wikitext")},
        initial_job_id="initial", initial_output=str(tmp_path / "initial.json"),
        work_per_inference={"corpora": {c: dict(chunks=2, prediction_tokens=4, raw_input_tokens=6) for c in ("target", "c4", "wikitext")}})
    jobs = [dict(id="initial", kind="initial", output=group["initial_output"], baseline_group="base", record_ids=[])]
    if alias:
        jobs.append(dict(id="shared", kind="adapted", baseline_group="base", weights_identity="checkpoint:weights",
            checkpoint="checkpoint_B4", checkpoint_sha256="weights", output=records[0]["output"], record_ids=[r["id"] for r in records]))
    else:
        jobs += [dict(id=r["group"], kind="adapted", baseline_group="base", weights_identity=r["weights_identity"],
            checkpoint=r["checkpoint"], checkpoint_sha256=r["checkpoint_sha256"], output=r["output"], record_ids=[r["id"]]) for r in records]
    manifest = dict(schema_version=2, mode="nll", phase_spec=str(spec_path), phase_spec_sha256=sha(spec_path),
        phases=[dict(name="cohort", role=role, model_tag="pythia-410m", data_family="main_v3", expected_phase=phase,
            rows=2, registry=str(registry_path), registry_sha256=sha(registry_path))], records=records, jobs=jobs, baseline_groups={"base": group})
    manifest_path = tmp_path / "manifest.json"; write(manifest_path, manifest)
    def result(checkpoint, values):
        corpora = {}
        for name, nll in zip(("target", "c4", "wikitext"), values):
            corpora[name] = dict(complete_corpus=True, chunks_evaluated=2, chunks_available=2, valid_prediction_tokens=4,
                raw_input_tokens=6, model_weight_dtype="torch.float32", nll=nll, sum_nll=nll * 4, kl=0., sum_kl=0.,
                kl_direction="teacher||student", kl_vocabulary="exact_full_vocabulary", kl_arithmetic="float32", precision="bf16",
                chunks=[dict(doc_id="d" + str(i), chunk_index=i, valid_prediction_tokens=2, sum_nll=nll * 2, sum_kl=0.) for i in range(2)])
        execution = {k: policy[k] for k in EXECUTION_KEYS}; execution["attention_backend_requested"] = "auto"
        return dict(model="base-model", revision=policy["revision"], precision="bf16", checkpoint=checkpoint, execution_provenance=execution, corpora=corpora)
    def attested(path, value, job_id, checkpoint_sha, weights_identity, logical=None):
        write(Path(path), value)
        att = dict(manifest_sha256=sha(manifest_path), job_id=job_id, verified_frozen_inputs=True, baseline_group="base",
            checkpoint_sha256=checkpoint_sha, weights_identity=weights_identity, output_sha256={str(path): sha(path)},
            inference_alias=logical is not None, inference_subprocess_wall_seconds=0 if logical else 7.)
        if logical:
            att.update(logical_record_id=logical, canonical_inference_subprocess_wall_seconds=7.)
        write(Path(path).with_name(Path(path).stem + "_execution.json"), att)
    attested(group["initial_output"], result(None, [4., 3., 2.]), "initial", None, None)
    for i, r in enumerate(records):
        values = [3., 3.1, 2.2] if alias or i == 0 else [2.5, 3.05, 2.1]
        value = result(r["checkpoint"], values)
        if alias and i == 1:
            value["inference_reuse"] = dict(canonical_output=records[0]["output"], canonical_job_id="shared", checkpoint_sha256="weights", canonical_checkpoint_sha256="weights", additional_model_forward_calls=0)
        attested(r["output"], value, r["inference_job_id"], r["checkpoint_sha256"], r["weights_identity"], r["id"] if alias and i == 1 else None)
    kwargs = dict(role=role, phase_name="cohort", model_tag="pythia-410m", data_family="main_v3", checkpoint=kind)
    return manifest_path, kwargs, manifest


def test_prepare_is_sealed_and_hand_calculated_metrics_keep_best_exposure(tmp_path, monkeypatch):
    path, kwargs, manifest = fixture(tmp_path, best=True)
    plan = prepare_plan(path, sha(path), **kwargs)
    assert plan["score_files_read"] is False
    with pytest.raises(PermissionError, match="sealed"):
        collect_scores(plan)
    bundle = collect_scores(plan, read_heldout_scores=True, n_bootstrap=20)
    first, second = bundle["endpoints"]
    assert first["g_new"] == 1.
    assert first["f_old_c4"] == pytest.approx(.1)
    assert second["g_new"] == 1.5
    assert bundle["paired_effects"][0]["g_new"] == .5
    assert first["target_prediction_tokens"] == 100
    assert first["full_training_target_prediction_tokens"] == 1000
    assert bundle["document_bootstrap"][0]["scope"].startswith("conditional")
    assert bundle["document_bootstrap"][0]["ratio"] == .05
    assert bundle["document_bootstrap"][0]["weight"] == .05
    assert bundle["document_bootstrap"][0]["run_a"] == "M"
    assert bundle["document_bootstrap"][0]["run_b"] == "B4"
    assert bundle["document_bootstrap"][0]["target_tokens_a"] == 100
    assert bundle["document_bootstrap"][0]["replay_tokens_a"] == 5
    assert bundle["document_bootstrap"][0]["actual_run_id_a"] == "M"
    assert bundle["endpoint_document_bootstrap"][0]["replay_prediction_tokens"] == 5
    assert bundle["endpoint_document_bootstrap"][0]["full_training_replay_prediction_tokens"] == 50
    assert bundle["endpoint_document_bootstrap"][0]["token_weighted"]["estimate"] == 1.
    # Delete all score paths: planning still succeeds and never probes them.
    for row in manifest["records"]:
        Path(row["output"]).unlink()
    prepare_plan(path, sha(path), **kwargs)


def test_first_real_score_read_callback_runs_before_json_parse(tmp_path,monkeypatch):
    import replay_study.nll_v2_analysis as analysis
    path,kwargs,manifest=fixture(tmp_path)
    plan=prepare_plan(path,sha(path),**kwargs)
    called=[];parsed=[];original=analysis.read
    scores={manifest["baseline_groups"]["base"]["initial_output"]}|{r["output"] for r in manifest["records"]}
    def before(path):called.append(str(path))
    def observed(path):
        if str(path) in scores:
            assert str(path) in called
            parsed.append(str(path))
        return original(path)
    monkeypatch.setattr(analysis,"read",observed)
    collect_scores(plan,read_heldout_scores=True,n_bootstrap=0,before_score_read=before)
    assert parsed[0]==manifest["baseline_groups"]["base"]["initial_output"] and set(parsed)==scores


def test_sha_attestation_and_complete_endpoint_rejections(tmp_path):
    path, kwargs, manifest = fixture(tmp_path)
    plan = prepare_plan(path, sha(path), **kwargs)
    output = Path(manifest["records"][0]["output"])
    original = output.read_text(); output.write_text(original + " ")
    with pytest.raises(ValueError, match="SHA"):
        collect_scores(plan, read_heldout_scores=True, n_bootstrap=0)
    output.write_text(original)
    output.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        collect_scores(plan, read_heldout_scores=True, n_bootstrap=0)
    partial = collect_scores(plan, read_heldout_scores=True, allow_partial=True, n_bootstrap=0)
    assert partial["complete"] is False and partial["completed_conditions"] == 1


def test_wrong_model_role_or_data_cannot_be_relabelled(tmp_path):
    path, kwargs, _ = fixture(tmp_path)
    for replacement in [dict(model_tag="pythia-1b"), dict(data_family="extended_v1"), dict(role="supplement_A")]:
        with pytest.raises(ValueError):
            prepare_plan(path, sha(path), **(kwargs | replacement))


def test_equal_role_labels_cannot_pool_different_baseline_data_identities(tmp_path):
    path, kwargs, manifest = fixture(tmp_path)
    manifest["baseline_groups"]["another"] = copy.deepcopy(manifest["baseline_groups"]["base"])
    manifest["baseline_groups"]["another"]["id"] = "another"
    manifest["records"][1]["baseline_group"] = "another"
    write(path, manifest)
    with pytest.raises(ValueError, match="one exact baseline"):
        prepare_plan(path, sha(path), **kwargs)


def test_alias_preserves_logical_identity_and_counts_one_physical_cost(tmp_path):
    path, kwargs, _ = fixture(tmp_path, role="supplement_A", alias=True)
    result = collect_scores(prepare_plan(path, sha(path), **kwargs), read_heldout_scores=True, n_bootstrap=0)
    assert [r["run_id"] for r in result["endpoints"]] == ["B4", "M"]
    assert {r["actual_run_id"] for r in result["endpoints"]} == {"physical_B4", "physical_M"}
    assert len(result["actual_training_costs"]) == 2
    assert all(r["incremental_phase_training_cost_factor"] == 0 for r in result["actual_training_costs"])
    assert len(result["physical_inference_costs"]) == 1
    assert result["physical_inference_costs"][0]["incremental_inference_subprocess_wall_seconds"] == 7.


def test_B_draw_seed_is_not_a_training_seed_and_best_is_forbidden(tmp_path):
    path, kwargs, _ = fixture(tmp_path, role="supplement_B")
    result = collect_scores(prepare_plan(path, sha(path), **kwargs), read_heldout_scores=True, n_bootstrap=0)
    assert result["replicate_unit"] == "selection_seed"
    assert result["paired_effects"][0]["selection_seed"] == 1001
    assert "seed" not in result["endpoints"][0]
    assert result["endpoints"][0]["training_seed"] == 11
    assert result["endpoint_statistics"][0]["bank_chunks"] == 20
    with pytest.raises(ValueError, match="final"):
        prepare_plan(path, sha(path), **(kwargs | dict(checkpoint="best")))


def test_valid_SHA_does_not_allow_changed_execution_or_misaligned_chunks(tmp_path):
    path, kwargs, manifest = fixture(tmp_path)
    output = Path(manifest["records"][0]["output"])
    att_path = output.with_name(output.stem + "_execution.json")
    original = json.loads(output.read_text())
    changed = copy.deepcopy(original)
    changed["execution_provenance"]["allow_tf32_cudnn"] = not changed["execution_provenance"]["allow_tf32_cudnn"]
    write(output, changed)
    att = json.loads(att_path.read_text()); att["output_sha256"][str(output)] = sha(output); write(att_path, att)
    with pytest.raises(ValueError, match="execution"):
        collect_scores(prepare_plan(path, sha(path), **kwargs), read_heldout_scores=True, n_bootstrap=0)
    changed = copy.deepcopy(original)
    changed["corpora"]["target"]["chunks"][0]["chunk_index"] = 900
    write(output, changed)
    att["output_sha256"][str(output)] = sha(output); write(att_path, att)
    with pytest.raises(ValueError, match="chunk identities"):
        collect_scores(prepare_plan(path, sha(path), **kwargs), read_heldout_scores=True, n_bootstrap=0)
