"""Read-only, registry-bound scientific analysis of formal replay experiments.

No model, CUDA, or tensor artifact is loaded here. Test evaluation JSON files are
read only when an evaluation_root is explicitly supplied by the caller.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CORPORA = ("target", "c4", "wikitext")
PAIRS = (("M", "B4"), ("C", "B4"), ("B2", "B1"), ("B4", "B3"),
         ("B3", "B1"), ("B4", "B2"))
GROUPS = ("B0", "B1", "B2", "B3", "B4", "D", "M", "C")
LABELS = {"B0": "Target only", "B1": "C4 CE", "B2": "C4 KL", "B3": "Generated CE",
          "B4": "Generated KL", "D": "Deduplicated KL", "M": "NLL-stratified KL", "C": "Cluster KL"}
METRICS = ("g_new", "f_old_c4", "f_old_wikitext", "kl_target", "kl_c4", "kl_wikitext",
           "nll_target", "nll_c4", "nll_wikitext", "g_new_equal_document",
           "f_old_c4_equal_document", "f_old_wikitext_equal_document",
           "kl_target_equal_document", "kl_c4_equal_document", "kl_wikitext_equal_document")
CI_SCOPE = "conditional on the trained checkpoints; does not replace training-seed SD"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_number(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    """Sample SD (ddof=1), deliberately undefined for fewer than two seeds."""
    x = np.asarray([finite_number(v, "observation") for v in values], dtype=np.float64)
    return {"mean": float(x.mean()) if len(x) else None,
            "std_ddof1": float(x.std(ddof=1)) if len(x) > 1 else None, "n": len(x)}


def _count(metrics: Mapping[str, Any]) -> int:
    value = metrics.get("valid_prediction_tokens", metrics.get("prediction_tokens"))
    if value is None or int(value) != value or int(value) <= 0:
        raise ValueError("positive integer prediction-token count required")
    return int(value)


def document_totals(corpus: Mapping[str, Any], field: str = "sum_nll") -> dict[str, tuple[float, int]]:
    """Aggregate all chunks of a document; never treat chunks as documents."""
    chunks = corpus.get("chunks")
    if not chunks:
        raise ValueError("document statistics require nonempty chunks with doc_id")
    sums: dict[str, list[float | int]] = {}
    seen = set()
    for row in chunks:
        if "doc_id" not in row or row["doc_id"] is None:
            raise ValueError("every chunk must contain doc_id")
        doc = str(row["doc_id"])
        if "chunk_index" not in row:
            raise ValueError("every chunk must contain chunk_index for paired alignment")
        key = (doc, row["chunk_index"])
        if key in seen:
            raise ValueError(f"duplicate chunk identity {key}")
        seen.add(key)
        count = _count(row)
        value = finite_number(row[field], field)
        total = sums.setdefault(doc, [0.0, 0])
        total[0] += value
        total[1] += count
    return {key: (float(v[0]), int(v[1])) for key, v in sums.items()}


def _chunk_identity_counts(corpus: Mapping[str, Any]) -> dict[tuple[str, Any], int]:
    return {(str(r["doc_id"]), r["chunk_index"]): _count(r) for r in corpus["chunks"]}


def _validate_corpus(corpus: Mapping[str, Any], name: str) -> None:
    if corpus.get("complete_corpus") is not True:
        raise ValueError(f"{name}: complete_corpus must explicitly be true")
    count = _count(corpus)
    nll = finite_number(corpus["nll"], f"{name}.nll")
    if "kl" in corpus:
        finite_number(corpus["kl"], f"{name}.kl")
        if corpus.get("kl_direction", "teacher||student") != "teacher||student":
            raise ValueError(f"{name}: retention KL must have teacher||student direction")
    if "sum_nll" in corpus and not math.isclose(nll, float(corpus["sum_nll"]) / count, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{name}: NLL sum/count mismatch")
    if corpus.get("chunks"):
        docs = document_totals(corpus)
        if sum(x[1] for x in docs.values()) != count:
            raise ValueError(f"{name}: chunk counts disagree with corpus count")
        if not math.isclose(sum(x[0] for x in docs.values()) / count, nll, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"{name}: chunk NLL disagrees with corpus NLL")


def endpoint_metrics(corpora: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Gain is initial-target minus endpoint-target; forgetting is endpoint-old minus initial-old."""
    out: dict[str, Any] = {}
    for name in CORPORA:
        current, initial = corpora[name], baseline[name]
        _validate_corpus(current, name)
        _validate_corpus(initial, f"initial.{name}")
        if _count(current) != _count(initial):
            raise ValueError(f"{name}: baseline and endpoint cover different prediction-token counts")
        if current.get("precision") and initial.get("precision") and current["precision"] != initial["precision"]:
            raise ValueError(f"{name}: baseline and endpoint precision differ")
        out[f"nll_{name}"] = finite_number(current["nll"], "nll")
        out[f"kl_{name}"] = finite_number(current["kl"], "kl") if "kl" in current else None
        out[f"kl_{name}_equal_document"] = None
        if current.get("chunks") and "kl" in current:
            kl_docs = document_totals(current, "sum_kl")
            out[f"kl_{name}_equal_document"] = float(np.mean([s/n for s,n in kl_docs.values()]))
        if current.get("chunks") and initial.get("chunks"):
            a, b = document_totals(current), document_totals(initial)
            if set(a) != set(b) or any(a[d][1] != b[d][1] for d in a):
                raise ValueError(f"{name}: baseline and endpoint document sets/counts differ")
            if _chunk_identity_counts(current) != _chunk_identity_counts(initial):
                raise ValueError(f"{name}: baseline and endpoint chunk identities/counts differ")
            delta_doc = float(np.mean([a[d][0] / a[d][1] - b[d][0] / b[d][1] for d in sorted(a)]))
        else:
            delta_doc = None
        delta = out[f"nll_{name}"] - float(initial["nll"])
        metric = "g_new" if name == "target" else f"f_old_{name}"
        out[metric] = -delta if name == "target" else delta
        out[f"{metric}_equal_document"] = (-delta_doc if name == "target" else delta_doc) if delta_doc is not None else None
    return out


def paired_document_bootstrap(corpus_a: Mapping[str, Any], corpus_b: Mapping[str, Any],
                              *, field: str = "sum_nll", n_bootstrap: int = 5000,
                              seed: int = 20261106, confidence: float = .95) -> dict[str, Any]:
    """Percentile CIs for A-B with the same resampled documents in both models.

    Token weighting is recomputed as a ratio of sums on each replicate; equal
    document weighting averages document-level means. The training seeds are
    never pooled or resampled by this function.
    """
    if n_bootstrap < 2 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap>=2 and 0<confidence<1 required")
    a, b = document_totals(corpus_a, field), document_totals(corpus_b, field)
    if set(a) != set(b):
        raise ValueError("paired bootstrap requires identical document sets")
    if _chunk_identity_counts(corpus_a) != _chunk_identity_counts(corpus_b):
        raise ValueError("paired bootstrap requires identical chunk identities and token counts")
    docs = sorted(a)
    counts = np.asarray([a[d][1] for d in docs], dtype=np.float64)
    if any(a[d][1] != b[d][1] for d in docs):
        raise ValueError("paired bootstrap requires equal within-document token counts")
    delta = np.asarray([a[d][0] - b[d][0] for d in docs], dtype=np.float64)
    means = delta / counts
    rng = np.random.default_rng(seed)
    weighted, equal = [], []
    for start in range(0, n_bootstrap, 256):
        draw = rng.integers(0, len(docs), size=(min(256, n_bootstrap - start), len(docs)))
        weighted.extend((delta[draw].sum(axis=1) / counts[draw].sum(axis=1)).tolist())
        equal.extend(means[draw].mean(axis=1).tolist())
    alpha = (1 - confidence) / 2
    def result(point, replicates):
        lo, hi = np.quantile(replicates, [alpha, 1 - alpha])
        return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi)}
    return {"token_weighted": result(delta.sum() / counts.sum(), weighted),
            "equal_document": result(means.mean(), equal), "documents": len(docs),
            "prediction_tokens": int(counts.sum()), "bootstrap_replicates": n_bootstrap,
            "bootstrap_seed": seed, "confidence": confidence, "field": field,
            "sign": "A minus B", "unit": "document with all its chunks", "scope": CI_SCOPE}


def aggregate_endpoints(rows: Sequence[Mapping[str, Any]], registry: Sequence[Mapping[str, Any]],
                        metrics: Sequence[str] = METRICS) -> list[dict[str, Any]]:
    expected: dict[tuple, set[int]] = defaultdict(set)
    actual: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in registry:
        expected[(row["group"], float(row["ratio"]), float(row["weight"]))].add(int(row["seed"]))
    for row in rows:
        key = (row["group"], float(row["ratio"]), float(row["weight"]))
        if key not in expected or int(row["seed"]) not in expected[key]:
            raise ValueError("endpoint outside the explicitly supplied registry")
        actual[(row["group"], float(row["ratio"]), float(row["weight"]))].append(row)
    out = []
    for (group, ratio, weight), seeds in sorted(expected.items()):
        current = actual[(group, ratio, weight)]
        if len({r["seed"] for r in current}) != len(current):
            raise ValueError("duplicate seed within group/b/weight")
        for metric in metrics:
            values = [r[metric] for r in current if r.get(metric) is not None]
            stats = mean_std(values)
            out.append({"group": group, "ratio": ratio, "weight": weight, "metric": metric,
                        **stats, "expected_n": len(seeds), "complete": stats["n"] == len(seeds)})
    return out


def paired_seed_differences(rows: Sequence[Mapping[str, Any]], registry: Sequence[Mapping[str, Any]]) -> tuple[list, list]:
    """Paired same-seed A-B effects at matching b and lambda; no unpaired subtraction."""
    lookup = {(r["group"], int(r["seed"]), float(r["ratio"]), float(r["weight"])): r for r in rows}
    expected = {(r["group"], int(r["seed"]), float(r["ratio"]), float(r["weight"])) for r in registry}
    if not set(lookup).issubset(expected):
        raise ValueError("paired endpoint outside the explicitly supplied registry")
    if len(lookup) != len(rows):
        raise ValueError("duplicate endpoint identity")
    differences, aggregates = [], []
    pair_settings = sorted({(a, b, ratio, weight) for a, b in PAIRS
                            for group, seed, ratio, weight in expected
                            if group == a and (b, seed, ratio, weight) in expected})
    for a, b, ratio, weight in pair_settings:
        expected_seeds = sorted(seed for group, seed, r, w in expected
                                if group == a and r == ratio and w == weight and (b, seed, r, w) in expected)
        for seed in expected_seeds:
            ra, rb = lookup.get((a, seed, ratio, weight)), lookup.get((b, seed, ratio, weight))
            if ra is None or rb is None:
                continue
            if ra.get("checkpoint") != rb.get("checkpoint"):
                raise ValueError("paired analysis cannot mix final and validation-selected best checkpoints")
            d = {"comparison": f"{a}-{b}", "seed": seed, "ratio": ratio, "weight": weight,
                 "run_a": ra["run_id"], "run_b": rb["run_id"], "sign": "A minus B", "checkpoint":ra.get("checkpoint")}
            if "target_prediction_tokens" in ra and "target_prediction_tokens" in rb:
                d.update(checkpoint_target_tokens_a=ra["target_prediction_tokens"], checkpoint_target_tokens_b=rb["target_prediction_tokens"],
                         equal_checkpoint_target_exposure=ra["target_prediction_tokens"] == rb["target_prediction_tokens"])
            for metric in METRICS:
                d[metric] = ra[metric] - rb[metric] if ra.get(metric) is not None and rb.get(metric) is not None else None
            differences.append(d)
        matched = [r for r in differences if r["comparison"] == f"{a}-{b}" and r["ratio"] == ratio and r["weight"] == weight]
        for metric in METRICS:
            stats = mean_std(r[metric] for r in matched if r.get(metric) is not None)
            aggregates.append({"comparison": f"{a}-{b}", "ratio": ratio, "weight": weight, "metric": metric,
                               **stats, "expected_n": len(expected_seeds), "complete": stats["n"] == len(expected_seeds),
                               "checkpoint":matched[0].get("checkpoint") if matched else None,
                               "all_pairs_equal_checkpoint_exposure":all(r.get("equal_checkpoint_target_exposure",True) for r in matched) if matched else None})
    return differences, aggregates


def training_costs(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Parent costs are additive; teacher/KL sub-timers remain diagnostic columns."""
    counters = summary["counters"]
    parent_keys = ("setup_seconds", "training_seconds", "validation_seconds", "checkpoint_seconds")
    result = {key: finite_number(counters.get(key, 0), key) for key in parent_keys}
    result["parent_component_sum_seconds"] = sum(result.values())
    result["total_accounted_wall_seconds"] = counters.get("total_accounted_wall_seconds")
    result["elapsed_seconds"] = counters.get("elapsed_seconds")
    total = result["total_accounted_wall_seconds"]
    result["unclassified_wall_seconds"] = float(total) - result["parent_component_sum_seconds"] if total is not None else None
    result["parent_components_exceed_reported_total"] = (result["unclassified_wall_seconds"] is not None and
                                                         result["unclassified_wall_seconds"] < -1e-3)
    for key in ("training_teacher_forward_seconds", "validation_teacher_forward_seconds", "validation_kl_arithmetic_seconds"):
        result[f"{key}_nested"] = counters.get(key)
    result["nested_timers_additive"] = False
    result["cost_scope"] = "full training run, even when an earlier best checkpoint is evaluated"
    result["recovered_cost_may_exclude_uncommitted_work"] = bool(summary.get("config", {}).get("resume_from"))
    return result


def shared_preparation_costs(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Formal pools only, once per seed, with nested clustering marked nonadditive."""
    out = []
    for seed, pool in sorted(summary.get("pools", {}).items()):
        if pool.get("role") != "formal":
            continue
        def append(stage, seconds, additive=True):
            out.append({"seed": int(seed), "stage": stage, "wall_seconds": finite_number(seconds, stage),
                        "additive": additive, "scope": "once per seed; shared across generated methods and all b",
                        "phase": "formal"})
        append("generation", pool["generation"]["generation_wall_seconds"])
        append("joint_teacher_nll_and_features", pool["teacher_statistics"]["wall_seconds"])
        for method, bank in sorted(pool.get("banks", {}).items()):
            append(f"selection_{method}", bank["wall_seconds"])
            if "clustering_wall_seconds" in bank:
                append("clustering_nested_in_selection_cluster", bank["clustering_wall_seconds"], False)
    return out


def trajectory_statistics(rows: Sequence[Mapping[str, Any]], registry: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected: dict[tuple, set[int]] = defaultdict(set)
    for row in registry:
        expected[(row["group"], row["ratio"], row["weight"])].add(int(row["seed"]))
    points: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        points[(row["group"], row["ratio"], row["weight"], row["target_prediction_tokens"])].append(row)
    out = []
    for (group, ratio, weight, exposure), current in sorted(points.items()):
        if len({r["seed"] for r in current}) != len(current):
            raise ValueError("duplicate seed/exposure in validation trajectory; check resumed event history")
        for metric in METRICS:
            stats = mean_std(r[metric] for r in current if r.get(metric) is not None)
            out.append({"group": group, "ratio": ratio, "weight": weight, "target_prediction_tokens": exposure,
                        "metric": metric, **stats, "expected_n": len(expected[(group, ratio, weight)]),
                        "complete": stats["n"] == len(expected[(group, ratio, weight)])})
    return out


def identical_bank_controls(rows: Sequence[Mapping[str, Any]], summaries: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep D-B4 as a diagnostic when deduplication changes no actual tokens."""
    lookup = {(r["group"], r["seed"], r["ratio"], r["weight"]):r for r in rows}
    out = []
    for row in rows:
        if row["group"] != "D":
            continue
        other = lookup.get(("B4", row["seed"], row["ratio"], row["weight"]))
        if other is None:
            continue
        a, b = summaries[row["run_id"]], summaries[other["run_id"]]
        ahash = ((a.get("provenance") or {}).get("replay") or {}).get("sha256_tensors")
        bhash = ((b.get("provenance") or {}).get("replay") or {}).get("sha256_tensors")
        same = ahash is not None and ahash == bhash
        result = {"comparison":"D-B4", "seed":row["seed"], "ratio":row["ratio"], "weight":row["weight"],
                  "run_a":row["run_id"], "run_b":other["run_id"], "same_replay_tensor_sha256":same,
                  "replay_tensor_sha256":ahash if same else None,
                  "same_source_sha256":a.get("provenance",{}).get("source_sha256")==b.get("provenance",{}).get("source_sha256"),
                  "interpretation":"identical-bank consistency control; no deduplication content intervention" if same else "banks differ"}
        for metric in METRICS:
            result[metric] = row[metric]-other[metric] if row.get(metric) is not None and other.get(metric) is not None else None
        out.append(result)
    return out


def matched_validation_loss(trajectory_a: Sequence[Mapping[str, Any]], trajectory_b: Sequence[Mapping[str, Any]],
                            *, initial_target_nll: float, n_levels: int = 5) -> dict[str, Any]:
    """First downward crossings in a common *validation* loss range.

    Piecewise-linear interpolation describes recorded validation curves, not a
    checkpoint, model interpolation, equal-budget result, or test selection.
    Later crossings on nonmonotone curves never replace the first crossing.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be positive")
    initial = finite_number(initial_target_nll, "initial_target_nll")
    measures = ("f_old_c4", "f_old_wikitext", "kl_target", "kl_c4", "kl_wikitext")
    def prepare(rows):
        points = sorted(rows, key=lambda r:r["target_prediction_tokens"])
        if not points:
            raise ValueError("matched loss requires validation trajectories for both runs")
        if len({p["target_prediction_tokens"] for p in points}) != len(points):
            raise ValueError("validation trajectory has duplicate target exposures")
        if any(p.get("split") == "test" for p in points):
            raise ValueError("matched loss may use validation trajectories only")
        if points[0]["target_prediction_tokens"] < 0:
            raise ValueError("negative target exposure")
        if points[0]["target_prediction_tokens"] > 0:
            points = [{"target_prediction_tokens":0, "replay_prediction_tokens":0,
                       "step":0,"nll_target":initial, **{m:0.0 for m in measures}}] + points
        elif not math.isclose(float(points[0]["nll_target"]),initial,rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError("zero-exposure validation point differs from initial baseline")
        losses = [finite_number(p["nll_target"], "target validation NLL") for p in points]
        return points, losses
    a, loss_a = prepare(trajectory_a); b, loss_b = prepare(trajectory_b)
    low, high = max(min(loss_a), min(loss_b)), initial
    result = {"common_loss_low":low,"common_loss_high":high,
              "nonmonotonic_a":any(y>x for x,y in zip(loss_a,loss_a[1:])),
              "nonmonotonic_b":any(y>x for x,y in zip(loss_b,loss_b[1:])),
              "source_split":"validation_only", "policy":"first downward crossing; piecewise-linear diagnostic",
              "actual_checkpoint":False,"equal_budget":False,"extrapolation":False,"points":[]}
    if not low < high:
        result["status"]="no_common_improved_loss_range"
        return result
    def crossing(points, level):
        choices = []
        for left,right in zip(points,points[1:]):
            u,v=float(left["nll_target"]),float(right["nll_target"])
            if not (u>v and v<=level<=u):
                continue
            fraction=(u-level)/(u-v)
            exposure=left["target_prediction_tokens"] + fraction*(right["target_prediction_tokens"]-left["target_prediction_tokens"])
            if choices and math.isclose(choices[-1]["interpolated_target_exposure"],exposure,abs_tol=1e-8):
                continue
            point={"interpolated_target_exposure":exposure,"bracket_target_low":left["target_prediction_tokens"],
                   "bracket_target_high":right["target_prediction_tokens"],"bracket_step_low":left.get("step"),
                   "bracket_step_high":right.get("step"),"fraction":fraction,
                   "bracket_target_loss_left":u,"bracket_target_loss_right":v}
            for metric in measures:
                point[metric] = (float(left[metric]) + fraction*(float(right[metric])-float(left[metric]))) if left.get(metric) is not None and right.get(metric) is not None else None
            choices.append(point)
        if not choices:
            raise ValueError("no measured bracket for a common loss level; extrapolation is forbidden")
        return {**choices[0],"downward_crossings":len(choices)}
    # Interior levels avoid the trivial initial point and a boundary-only comparison.
    for level in np.linspace(low,high,n_levels+2)[1:-1]:
        ca,cb=crossing(a,float(level)),crossing(b,float(level))
        point={"target_validation_nll":float(level),"actual_checkpoint":False,"equal_budget":False}
        for prefix,current in (("a",ca),("b",cb)):
            point.update({f"{key}_{prefix}":value for key,value in current.items()})
        for metric in measures:
            point[f"delta_{metric}"]=ca[metric]-cb[metric] if ca[metric] is not None and cb[metric] is not None else None
        result["points"].append(point)
    result["status"]="available_validation_diagnostic"
    return result


def matched_validation_comparisons(trajectories: Sequence[Mapping[str, Any]], registry: Sequence[Mapping[str, Any]],
                                   initial_target_nll: float) -> tuple[list, list]:
    by_run: dict[str,list] = defaultdict(list)
    for row in trajectories:
        if row["complete_training"]:
            by_run[row["run_id"]].append(row)
    entries={(r["group"],r["seed"],r["ratio"],r["weight"]):r for r in registry}
    pairs=[]
    for group_a,group_b in PAIRS:
        for key,a in sorted(entries.items()):
            group,seed,ratio,weight=key
            b=entries.get((group_b,seed,ratio,weight))
            if group==group_a and b is not None:
                pairs.append((a,b))
    for key,a in sorted(entries.items()):
        group,seed,ratio,weight=key
        b=entries.get(("B0",seed,0.,0.))
        if group in {"B1","B2","B3","B4","M","C"} and math.isclose(ratio,.05) and b is not None:
            pairs.append((a,b))
    ranges,points=[],[]
    for a,b in pairs:
        if a["id"] not in by_run or b["id"] not in by_run:
            continue
        result=matched_validation_loss(by_run[a["id"]],by_run[b["id"]],initial_target_nll=initial_target_nll)
        identity={"comparison":f"{a['group']}-{b['group']}","seed":a["seed"],"ratio_a":a["ratio"],"ratio_b":b["ratio"],
                  "weight_a":a["weight"],"weight_b":b["weight"],"run_a":a["id"],"run_b":b["id"]}
        ranges.append({**identity,**{k:v for k,v in result.items() if k!="points"}})
        for point in result["points"]:
            points.append({**identity,"source_split":"validation_only", "nonmonotonic_a":result["nonmonotonic_a"],
                           "nonmonotonic_b":result["nonmonotonic_b"],**point})
    return ranges,points


def _registry(path: str | Path, analysis_status: str = "active") -> list[dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty formal registry list required")
    allowed = {"formal"} if analysis_status == "active" else {"formal", "provisional"}
    if any(r.get("phase") not in allowed for r in rows):
        raise ValueError("analysis registry may contain formal runs only; pilots must be separate")
    if analysis_status == "active" and any(r.get("active") is False or r.get("status") in {"provisional", "superseded", "invalid"} for r in rows):
        raise ValueError("active analysis cannot include provisional or superseded registry entries")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate run IDs in registry")
    keys = [(r["group"], int(r["seed"]), float(r["ratio"]), float(r["weight"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate scientific identities in registry")
    return rows


def _identity(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {"run_id": entry["id"], "group": entry["group"], "seed": int(entry["seed"]),
            "ratio": float(entry["ratio"]), "weight": float(entry["weight"])}


def _check_summary(summary: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
    config = summary["config"]
    for field, expected in (("seed", entry["seed"]), ("replay_ratio", entry["ratio"]),
                            ("replay_weight", entry["weight"]), ("learning_rate", entry.get("learning_rate"))):
        if expected is not None and config.get(field) != expected:
            raise ValueError(f"{entry['id']}: summary {field} disagrees with registry")
    label = config.get("run_label")
    if label is not None and label != entry["id"]:
        raise ValueError(f"{entry['id']}: run_label mismatch; refusing a pilot or different run")
    if entry.get("config_path") and Path(entry["config_path"]).exists():
        frozen = read_json(entry["config_path"])
        operational = {"output_dir", "device", "resume_from", "max_steps", "deadline_utc", "deadline_margin_seconds",
                       "save_every_steps", "keep_step_checkpoints"}
        for field in frozen:
            if field not in operational and config.get(field) != frozen[field]:
                raise ValueError(f"{entry['id']}: summary {field} disagrees with frozen config")


def selected_checkpoint_exposure(summary: Mapping[str, Any], checkpoint: str) -> dict[str, Any]:
    """A best-by-validation model can precede the fully paid training endpoint."""
    counters = summary["counters"]
    if checkpoint == "final":
        step, target, replay = counters["step"], counters["target_prediction_tokens"], counters["replay_prediction_tokens"]
    elif checkpoint == "best":
        selected = summary.get("best_target_validation")
        if not selected:
            raise ValueError("best checkpoint requires a saved validation selection record")
        matches = [p for p in summary.get("validation_trajectory", []) if p["step"] == selected["step"] and
                   p["target_prediction_tokens"] == selected["target_prediction_tokens"]]
        if len(matches) != 1:
            raise ValueError("best selection must match exactly one validation trajectory point")
        point = matches[0]
        if not math.isclose(float(selected["nll"]), float(point["metrics"]["target"]["nll"]), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("best selection NLL disagrees with its validation trajectory")
        step, target, replay = point["step"], point["target_prediction_tokens"], point["replay_prediction_tokens"]
    else:
        raise ValueError("checkpoint must be final or best")
    if target > counters["target_prediction_tokens"] or replay > counters["replay_prediction_tokens"]:
        raise ValueError("selected checkpoint exposure exceeds full training exposure")
    return {"checkpoint_step":step, "target_prediction_tokens":target, "replay_prediction_tokens":replay,
            "full_training_target_prediction_tokens":counters["target_prediction_tokens"],
            "full_training_replay_prediction_tokens":counters["replay_prediction_tokens"]}


def _event_trajectory(entry: Mapping[str, Any], sources: dict[str, str], initial_execution=None) -> list[dict[str, Any]]:
    """Progress only: never promote a live event to a complete endpoint."""
    path = Path(entry.get("output_dir", ".")) / "events.jsonl"
    if not path.exists():
        return []
    contents = path.read_bytes()
    sources[str(path)] = hashlib.sha256(contents).hexdigest()
    trajectory = []
    identity_seen = False
    start = None
    for line in contents.decode().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a writer can be in the middle of the last line
        if row.get("event") in {"start", "resume"}:
            _check_summary({"config":row["config"]},entry)
            identity_seen = True
            start = row
        if row.get("event") == "validation":
            if not identity_seen:
                raise ValueError(f"{entry['id']}: live validation events lack a verified start/resume identity")
            if initial_execution:
                verify_recorded_execution(initial_execution,training_validation_execution(
                    {"config":start["config"],"provenance":start.get("provenance",{}),"validation_trajectory":[row]}))
            trajectory.append(row)
    return trajectory


def _protocol_status(path: str | Path, registry_path: str | Path, analysis_status: str) -> dict[str, Any]:
    status_path=Path(path).resolve()
    status=read_json(status_path)
    base=status_path.parent.parent if status_path.parent.name=="docs" else status_path.parent
    def resolved(value):
        p=Path(value)
        return p.resolve() if p.is_absolute() else (base/p).resolve()
    if analysis_status=="active":
        active=status.get("active_registry")
        if not active:
            raise ValueError("protocol_status.json has no frozen active registry")
        if resolved(active)!=Path(registry_path).resolve():
            raise ValueError("supplied registry is not the active registry in protocol_status.json")
        if any(resolved(p)==Path(registry_path).resolve() for p in status.get("provisional_registries",[])):
            raise ValueError("active registry is also listed as provisional")
    return status


def verify_recorded_execution(initial: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Compare saved policy dictionaries; this performs no model/CUDA work."""
    from .execution import verify_matching_execution
    verify_matching_execution(initial,current)
    for field in ("model_weight_dtype","cudnn_sdp_enabled"):
        if field in initial and current.get(field) != initial[field]:
            raise ValueError(f"baseline/adapted execution policy differs: {field}")


def training_validation_execution(summary: Mapping[str, Any]) -> dict[str, Any]:
    config,provenance=summary["config"],summary.get("provenance",{})
    backends=provenance.get("sdpa_backends_enabled",{})
    dtypes={m.get("model_weight_dtype") for point in summary.get("validation_trajectory",[])
            for m in point["metrics"].values() if m.get("model_weight_dtype") is not None}
    if len(dtypes)>1:
        raise ValueError("validation trajectory mixes model weight dtypes")
    weight_dtype=provenance.get("model_weight_dtype")
    if weight_dtype is not None and dtypes and weight_dtype not in dtypes:
        raise ValueError("saved model weight dtype differs from validation trajectory")
    device=str(provenance.get("device",config.get("device",""))).split(":",1)[0]
    precision=provenance.get("precision",config.get("precision"))
    return {"torch_version":provenance.get("torch_version"),"cuda_version":provenance.get("cuda_version"),
            "precision":precision,"device_type":device or None,
            "effective_precision":"cuda_bfloat16_autocast" if device=="cuda" and precision=="bf16" else "float32",
            "gpu_name":provenance.get("gpu") if device=="cuda" else None,
            "attention_implementation":provenance.get("attention_implementation",config.get("attention_implementation")),
            "attention_backend_requested":provenance.get("attention_backend_policy"),
            "flash_sdp_enabled":backends.get("flash"),"mem_efficient_sdp_enabled":backends.get("memory_efficient"),
            "math_sdp_enabled":backends.get("math"),"cudnn_sdp_enabled":backends.get("cudnn"),
            "deterministic_algorithms":provenance.get("deterministic_algorithms"),
            "deterministic_warn_only":provenance.get("deterministic_warn_only"),
            "allow_tf32_matmul":provenance.get("allow_tf32_matmul",provenance.get("allow_tf32")),
            "allow_tf32_cudnn":provenance.get("allow_tf32_cudnn",provenance.get("allow_tf32")),
            "cudnn_deterministic":provenance.get("cudnn_deterministic"),"cudnn_benchmark":provenance.get("cudnn_benchmark"),
            "CUBLAS_WORKSPACE_CONFIG":provenance.get("cublas_workspace_config"),
            "model_weight_dtype":weight_dtype or (next(iter(dtypes)) if dtypes else None)}


def collect_analysis(registry_path: str | Path, baseline_path: str | Path, *,
                     evaluation_root: str | Path | None = None, checkpoint: str = "final",
                     evaluation_baseline: str | Path | None = None,
                     generation_summary: str | Path | None = None,
                     n_bootstrap: int = 5000, bootstrap_seed: int = 20261106,
                     analysis_status: str, protocol_status_path: str | Path | None = None,
                     require_execution_provenance: bool = False) -> dict[str, Any]:
    if checkpoint not in {"final", "best"}:
        raise ValueError("checkpoint must be final or best")
    if analysis_status not in {"active", "provisional-numerical-noise"}:
        raise ValueError("analysis_status must be active or provisional-numerical-noise")
    if evaluation_root and analysis_status != "active":
        raise ValueError("test evaluation JSON is reserved for an explicitly active, frozen registry")
    protocol_status=_protocol_status(protocol_status_path,registry_path,analysis_status) if protocol_status_path else None
    registry = _registry(registry_path, analysis_status)
    baseline_file = Path(evaluation_baseline or Path(evaluation_root) / "initial_pythia-410m.json") if evaluation_root else Path(baseline_path)
    baseline_payload = read_json(baseline_file)
    if baseline_payload.get("checkpoint") is not None:
        raise ValueError("initial baseline must be the unadapted model, without a fine-tuned checkpoint")
    baseline = baseline_payload["corpora"]
    validation_baseline_payload = read_json(baseline_path) if evaluation_root else baseline_payload
    validation_baseline = validation_baseline_payload["corpora"]
    initial_validation_execution=validation_baseline_payload.get("execution_provenance")
    if require_execution_provenance and (not initial_validation_execution or not baseline_payload.get("execution_provenance")):
        raise ValueError("active production analysis requires execution_provenance on validation and endpoint initial baselines")
    for corpus in CORPORA:
        _validate_corpus(baseline[corpus], f"initial.{corpus}")
        if "kl" in baseline[corpus] and abs(float(baseline[corpus]["kl"])) > 1e-8:
            raise ValueError("initial teacher self-KL must be zero; a different checkpoint may have been used")
    endpoint_rows, trajectory_rows, costs, observed_costs, availability = [], [], [], [], []
    evaluation_payloads, loaded_summaries = {}, {}
    sources = {str(Path(registry_path)): sha256(registry_path), str(baseline_file): sha256(baseline_file),
               str(Path(baseline_path)): sha256(baseline_path)}
    if protocol_status_path:
        sources[str(Path(protocol_status_path))]=sha256(protocol_status_path)
    for entry in registry:
        identity = _identity(entry)
        if entry.get("config_path") and Path(entry["config_path"]).exists():
            sources[str(Path(entry["config_path"]))] = sha256(entry["config_path"])
        path = Path(entry["summary_path"])
        summary = read_json(path) if path.exists() else None
        if summary is not None:
            loaded_summaries[entry["id"]] = summary
            sources[str(path)] = sha256(path)
            _check_summary(summary, entry)
            if analysis_status == "active" and summary.get("protocol_status") in {"provisional", "superseded", "invalid"}:
                raise ValueError(f"{entry['id']}: provisional summary cannot enter active analysis")
            if baseline_payload.get("revision") and summary["config"].get("model_revision") != baseline_payload["revision"]:
                raise ValueError(f"{entry['id']}: baseline/model revision mismatch")
            if validation_baseline_payload.get("precision") and summary["config"].get("precision") != validation_baseline_payload["precision"]:
                raise ValueError(f"{entry['id']}: validation precision differs from initial baseline")
            if analysis_status=="active" and initial_validation_execution and summary.get("validation_trajectory"):
                verify_recorded_execution(initial_validation_execution,training_validation_execution(summary))
        complete_training = (summary is not None and summary.get("status") == "completed" and
                             summary["counters"]["target_prediction_tokens"] == summary["config"]["target_prediction_tokens"])
        if complete_training:
            expected_replay = int(summary["config"]["target_prediction_tokens"] * Fraction(str(entry["ratio"])))
            if summary["counters"]["replay_prediction_tokens"] != expected_replay:
                raise ValueError(f"{entry['id']}: actual replay exposure differs from cumulative floor(b*T)")
        observed_cost = None
        if summary is not None:
            observed_cost={**identity, **training_costs(summary), "run_status":summary.get("status"),
                           "complete_training":complete_training,
                           "target_prediction_tokens":summary["counters"]["target_prediction_tokens"],
                           "replay_prediction_tokens":summary["counters"]["replay_prediction_tokens"],
                           "bank_unique_prediction_tokens":summary.get("replay_bank_unique_prediction_tokens"),
                           "peak_gpu_memory_bytes":summary.get("peak_gpu_memory_bytes")}
            observed_costs.append(observed_cost)
        trajectory = summary.get("validation_trajectory", []) if summary else _event_trajectory(
            entry,sources,initial_validation_execution if analysis_status=="active" else None)
        for point in trajectory:
            values = endpoint_metrics(point["metrics"], validation_baseline)
            trajectory_rows.append({**identity, "step": point["step"],
                                    "target_prediction_tokens": point["target_prediction_tokens"],
                                    "replay_prediction_tokens": point["replay_prediction_tokens"],
                                    "complete_training": complete_training, **values})
        endpoint = None
        reason = "missing_summary" if summary is None else "training_not_complete"
        if complete_training:
            costs.append(observed_cost)
            if evaluation_root:
                eval_path = Path(evaluation_root) / entry["id"] / f"{checkpoint}.json"
                reason = "missing_evaluation"
                if eval_path.exists():
                    payload = read_json(eval_path)
                    if payload.get("revision") != baseline_payload.get("revision"):
                        raise ValueError(f"{entry['id']}: evaluation revision differs from baseline")
                    if baseline_payload.get("precision") and payload.get("precision") != baseline_payload["precision"]:
                        raise ValueError(f"{entry['id']}: evaluation precision differs from initial baseline")
                    expected_checkpoint = (summary.get("checkpoint_path") if checkpoint == "final" else
                                           (summary.get("best_target_validation") or {}).get("checkpoint_path"))
                    if not payload.get("checkpoint") or not expected_checkpoint:
                        raise ValueError(f"{entry['id']}: evaluation and summary must identify the chosen checkpoint")
                    if Path(payload["checkpoint"]).resolve() != Path(expected_checkpoint).resolve():
                        raise ValueError(f"{entry['id']}: evaluation uses a different checkpoint than the frozen selection")
                    initial_execution=baseline_payload.get("execution_provenance")
                    if initial_execution:
                        verify_recorded_execution(initial_execution,payload.get("execution_provenance",{}))
                    endpoint = payload["corpora"]
                    evaluation_payloads[entry["id"]] = payload
                    sources[str(eval_path)] = sha256(eval_path)
            elif trajectory and trajectory[-1]["target_prediction_tokens"] == summary["config"]["target_prediction_tokens"]:
                endpoint = trajectory[-1]["metrics"]
            else:
                reason = "missing_final_exposure_validation"
        if endpoint is not None:
            exposure = selected_checkpoint_exposure(summary, checkpoint if evaluation_root else "final")
            endpoint_rows.append({**identity, "split": "test" if evaluation_root else "validation",
                                  "checkpoint": checkpoint if evaluation_root else "fixed_exposure_final",
                                  **exposure,
                                  **endpoint_metrics(endpoint, baseline)})
            reason = "ready"
        availability.append({**identity, "status": reason})
    paired_rows, paired_stats = paired_seed_differences(endpoint_rows, registry)
    if analysis_status != "active":
        paired_rows, paired_stats = [], []  # No scientific effect comparison from a numerically provisional batch.
    bootstrap_rows, bootstrap_endpoint_rows = [], []
    if evaluation_root:
        def append_bootstrap(destination, metadata, a, b, field, sign=1):
            result = paired_document_bootstrap(a, b, field=field, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
            for weighting in ("token_weighted", "equal_document"):
                v = result[weighting]
                lo, hi = (v["ci_low"], v["ci_high"]) if sign == 1 else (-v["ci_high"], -v["ci_low"])
                destination.append({**metadata, "weighting": weighting, "estimate": sign * v["estimate"],
                                    "ci_low": lo, "ci_high": hi, **{k: v for k, v in result.items()
                                                                 if k not in {"token_weighted", "equal_document", "sign"}}})
        for row in endpoint_rows:
            payload = evaluation_payloads[row["run_id"]]
            for corpus in CORPORA:
                metric = "g_new" if corpus == "target" else f"f_old_{corpus}"
                append_bootstrap(bootstrap_endpoint_rows, {**{k: row[k] for k in ("run_id", "group", "seed", "ratio", "weight")},
                                                          "corpus": corpus, "metric": metric, "checkpoint": checkpoint},
                                 payload["corpora"][corpus], baseline[corpus], "sum_nll", -1 if corpus == "target" else 1)
                if "kl" in payload["corpora"][corpus] and "kl" in baseline[corpus]:
                    append_bootstrap(bootstrap_endpoint_rows,
                                     {**{k: row[k] for k in ("run_id", "group", "seed", "ratio", "weight")},
                                      "corpus": corpus, "metric": f"kl_{corpus}", "checkpoint": checkpoint},
                                     payload["corpora"][corpus], baseline[corpus], "sum_kl")
        for pair in paired_rows:
            a, b = evaluation_payloads[pair["run_a"]], evaluation_payloads[pair["run_b"]]
            for corpus in CORPORA:
                for field in ("sum_nll", "sum_kl"):
                    if field == "sum_kl" and ("kl" not in a["corpora"][corpus] or "kl" not in b["corpora"][corpus]):
                        continue
                    append_bootstrap(bootstrap_rows, {**{k: pair[k] for k in ("comparison", "seed", "ratio", "weight", "run_a", "run_b")},
                                                     "corpus": corpus, "metric": "delta_nll" if field == "sum_nll" else "delta_kl",
                                                     "checkpoint": checkpoint, "sign": "A minus B"},
                                     a["corpora"][corpus], b["corpora"][corpus], field)
    shared, preparation_accounting = [], {}
    if generation_summary and Path(generation_summary).exists():
        preparation = read_json(generation_summary)
        shared = shared_preparation_costs(preparation)
        preparation_accounting = {"source_complete":preparation.get("complete"),
                                  "formal_additive_shared_seconds":sum(r["wall_seconds"] for r in shared if r["additive"]),
                                  "mixed_formal_and_pilot_orchestration_wall_seconds":preparation.get("orchestration_wall_seconds"),
                                  "mixed_formal_and_pilot_model_load_wall_seconds":preparation.get("model_load_wall_seconds"),
                                  "note":"Mixed-phase orchestration/model-loading overhead is retained but not attributed entirely to formal runs. Do not add nested clustering again."}
        sources[str(Path(generation_summary))] = sha256(generation_summary)
    shared_stats = []
    for stage in sorted({r["stage"] for r in shared}):
        current = [r for r in shared if r["stage"] == stage]
        shared_stats.append({"stage": stage, **mean_std(r["wall_seconds"] for r in current),
                             "sum_seconds": sum(r["wall_seconds"] for r in current),
                             "additive": current[0]["additive"], "expected_n": len({r["seed"] for r in registry})})
    cost_metrics = ("setup_seconds", "training_seconds", "validation_seconds", "checkpoint_seconds",
                    "parent_component_sum_seconds", "total_accounted_wall_seconds", "unclassified_wall_seconds",
                    "training_teacher_forward_seconds_nested", "validation_teacher_forward_seconds_nested",
                    "validation_kl_arithmetic_seconds_nested", "peak_gpu_memory_bytes")
    matched_ranges,matched_points=(matched_validation_comparisons(trajectory_rows,registry,float(validation_baseline["target"]["nll"]))
                                   if analysis_status=="active" else ([],[]))
    return {"registry": registry, "endpoints": endpoint_rows,
            "endpoint_statistics": aggregate_endpoints(endpoint_rows, registry),
            "paired_seed_differences": paired_rows, "paired_statistics": paired_stats,
            "identical_bank_controls": identical_bank_controls(endpoint_rows, loaded_summaries),
            "matched_validation_ranges":matched_ranges,"matched_validation_loss":matched_points,
            "trajectories": trajectory_rows, "trajectory_statistics": trajectory_statistics(trajectory_rows, registry),
            "training_costs": costs, "cost_statistics": aggregate_endpoints(costs, registry, cost_metrics),
            "observed_run_costs":observed_costs,
            "shared_preparation_costs": shared, "shared_preparation_statistics": shared_stats,
            "preparation_accounting":preparation_accounting,
            "protocol_status_snapshot":protocol_status,
            "availability": availability, "document_bootstrap_endpoints": bootstrap_endpoint_rows,
            "paired_document_bootstrap": bootstrap_rows,
            "metadata": {"generated_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(), "analysis_source_sha256":sha256(__file__),
                         "split": "test" if evaluation_root else "validation", "checkpoint": checkpoint if evaluation_root else "fixed_exposure_final",
                         "checkpoint_exposure_policy":"validation-selected best can have different exposure; full training cost remains charged" if evaluation_root and checkpoint == "best" else "fixed full target-token exposure",
                         "complete": len(endpoint_rows) == len(registry) and analysis_status == "active",
                         "analysis_status":analysis_status, "scientific_comparisons_suppressed":analysis_status != "active",
                         "protocol_status_gate_applied":protocol_status is not None,
                         "execution_provenance_required":require_execution_provenance,
                         "completed_endpoints": len(endpoint_rows),
                         "expected_endpoints": len(registry), "formal_only": analysis_status == "active",
                         "pilot_results_excluded": True, "model_revision":baseline_payload.get("revision"), "sources_sha256": sources,
                         "signs": {"g_new": "initial target NLL minus endpoint target NLL; higher is better",
                                   "f_old": "endpoint old-domain NLL minus initial NLL; lower is better",
                                   "paired": "A minus B; gain positive favors A, forgetting/KL negative favors A"},
                         "seed_sd": "sample SD with ddof=1; undefined for n<2",
                         "document_bootstrap_scope": CI_SCOPE,
                         "costs": "teacher/KL nested timers are nonadditive; shared preparation charged once per seed"}}


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["no_rows"])
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(bundle: Mapping[str, Any], output: str | Path, *, allow_partial: bool = False,
                   plots: bool = True) -> None:
    if not bundle["metadata"]["complete"] and not allow_partial:
        raise ValueError("formal results are incomplete; pass --allow-partial for explicitly labelled provisional output")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("endpoints", "endpoint_statistics", "paired_seed_differences", "paired_statistics", "trajectories",
                 "trajectory_statistics", "training_costs", "cost_statistics", "shared_preparation_costs",
                 "shared_preparation_statistics", "availability", "document_bootstrap_endpoints", "paired_document_bootstrap", "identical_bank_controls", "observed_run_costs"):
        write_csv(output / f"{name}.csv", bundle[name])
    for name in ("matched_validation_ranges","matched_validation_loss"):
        write_csv(output / f"{name}.csv",bundle[name])
    budget = [r for r in bundle["endpoint_statistics"] if r["group"] in {"B4", "M"}]
    write_csv(output / "budget_curves.csv", budget)
    (output / "analysis_manifest.json").write_text(json.dumps(bundle["metadata"], ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "preparation_accounting.json").write_text(json.dumps(bundle["preparation_accounting"], ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if bundle["protocol_status_snapshot"] is not None:
        (output/"protocol_status_snapshot.json").write_text(json.dumps(bundle["protocol_status_snapshot"],ensure_ascii=False,indent=2)+"\n")
    if plots:
        plot_analysis(bundle, output)


def plot_analysis(bundle: Mapping[str, Any], output: str | Path) -> None:
    """Publication exports with seed dots, means, and SD; never imply a missing SD is zero."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    output = Path(output)
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "savefig.dpi": 320})
    palette = ("#666666", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#999999", "#D55E00", "#CC79A7")
    colors = dict(zip(GROUPS, palette))
    markers = dict(zip(GROUPS, ("o", "s", "^", "v", "D", "P", "X", "*")))
    meta = bundle["metadata"]
    partial = not meta["complete"]
    status_title = "PROVISIONAL: numerical noise" if meta.get("analysis_status") != "active" else ("PARTIAL" if partial else "COMPLETE")
    title = f"{meta['split'].title()} | {status_title}: {meta['completed_endpoints']}/{meta['expected_endpoints']} runs"
    if meta["checkpoint"] == "best":
        title += "\nBest by target validation; checkpoint exposures vary"
    def finish(fig, name):
        fig.suptitle(title, fontsize=11, color="#9C4B00" if partial else "black")
        fig.tight_layout(rect=(0, .03, 1, .94))
        for extension in ("png", "pdf", "svg"):
            fig.savefig(figure_dir / f"{name}.{extension}", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    def empty(ax):
        ax.text(.5, .5, "No completed formal endpoints", ha="center", va="center", transform=ax.transAxes, color=".45")
    main = [r for r in bundle["endpoints"] if r["group"] == "B0" or math.isclose(r["ratio"], .05)]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for ax, name in zip(axes, ("C4", "WikiText-103")):
        ax.set_title(name); ax.set_xlabel(r"New-domain gain $G_{new}$ (nats/token)")
        ax.set_ylabel(r"Old-domain forgetting $F_{old}$ (nats/token)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.axhline(0, color=".75", linewidth=.6); ax.grid(alpha=.18)
    handles = []
    for group in GROUPS:
        rows = [r for r in main if r["group"] == group]
        if not rows:
            continue
        handles.append(Line2D([0], [0], marker=markers[group], color=colors[group], linestyle="none",
                              label=f"{group}: {LABELS[group]} (n={len(rows)})"))
        for ax, corpus, name in zip(axes, ("c4", "wikitext"), ("C4", "WikiText-103")):
            x = np.array([r["g_new"] for r in rows]); y = np.array([r[f"f_old_{corpus}"] for r in rows])
            ax.scatter(x, y, s=18, marker=markers[group], color=colors[group], alpha=.42, linewidths=.6)
            ax.errorbar(x.mean(), y.mean(), xerr=x.std(ddof=1) if len(x)>1 else None,
                        yerr=y.std(ddof=1) if len(y)>1 else None, marker=markers[group],
                        color=colors[group], capsize=3, markersize=7, linewidth=1.2, linestyle="none")
    if not main:
        for ax in axes: empty(ax)
    if handles: fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(.5, -.10))
    finish(fig, "gain_forgetting_scatter")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, metric, label in zip(axes, ("g_new", "f_old_c4", "f_old_wikitext"),
                                  ("New-domain gain", "C4 forgetting", "WikiText-103 forgetting")):
        any_data = False
        for group in ("B4", "M"):
            summaries = [r for r in bundle["endpoint_statistics"] if r["group"] == group and r["metric"] == metric and r["n"]]
            summaries.sort(key=lambda r:r["ratio"])
            if not summaries: continue
            any_data = True
            for r in bundle["endpoints"]:
                if r["group"] == group: ax.scatter(r["ratio"], r[metric], s=14, color=colors[group], alpha=.3)
            ax.plot([r["ratio"] for r in summaries], [r["mean"] for r in summaries], color=colors[group],
                    marker=markers[group], label=group)
            for r in summaries:
                if r["std_ddof1"] is not None: ax.errorbar(r["ratio"], r["mean"], yerr=r["std_ddof1"], color=colors[group], capsize=3)
        if not any_data: empty(ax)
        ax.set_xscale("log"); ax.set_xticks([.01,.05,.1], ["0.01","0.05","0.10"])
        ax.set_xlabel(r"Replay ratio $b=T_R/T_{new}$"); ax.set_ylabel("nats/token"); ax.set_title(label); ax.grid(alpha=.18)
    if axes[0].get_legend_handles_labels()[0]: axes[0].legend(frameon=False)
    finish(fig, "budget_curves")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, metric, label in zip(axes, ("g_new", "f_old_c4", "f_old_wikitext"),
                                  ("New-domain gain", "C4 forgetting", "WikiText-103 forgetting")):
        for group in GROUPS:
            rows = [r for r in bundle["trajectories"] if r["group"] == group and (group == "B0" or math.isclose(r["ratio"], .05))]
            for seed in sorted({r["seed"] for r in rows}):
                current = sorted([r for r in rows if r["seed"] == seed], key=lambda r:r["target_prediction_tokens"])
                ax.plot([r["target_prediction_tokens"]/1e6 for r in current], [r[metric] for r in current],
                        color=colors[group], alpha=.22, linewidth=.65)
            points = defaultdict(list)
            for row in rows: points[row["target_prediction_tokens"]].append(row[metric])
            if points:
                x = sorted(points)
                ax.plot(np.array(x)/1e6, [np.mean(points[k]) for k in x], color=colors[group], label=group, linewidth=1.3)
        if not bundle["trajectories"]: empty(ax)
        ax.set_title(label); ax.set_xlabel("Target prediction-token exposure (million)"); ax.set_ylabel("nats/token"); ax.grid(alpha=.18)
    if axes[0].get_legend_handles_labels()[0]: axes[0].legend(ncol=4, fontsize=7, frameon=False)
    # These are always validation trajectories, including in a later test report.
    old_split = meta["split"]
    if old_split == "test": title = title.replace("Test", "Validation trajectories")
    finish(fig, "validation_trajectories")
    if old_split == "test": title = title.replace("Validation trajectories", "Test")

    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    costrows = [r for r in bundle["training_costs"] if r["group"] == "B0" or math.isclose(r["ratio"], .05)]
    present = [g for g in GROUPS if any(r["group"]==g for r in costrows)]
    bottom = np.zeros(len(present))
    for field, label, color in (("setup_seconds","Setup","#BDBDBD"),("training_seconds","Training (teacher included)","#0072B2"),
                                ("validation_seconds","Validation (teacher/KL included)","#56B4E9"),("checkpoint_seconds","Checkpoint I/O","#E69F00")):
        values = [np.mean([r[field] for r in costrows if r["group"]==g]) for g in present]
        ax.bar(present, values, bottom=bottom, label=label, color=color, width=.7); bottom += values
    if not present: empty(ax)
    ax.set_ylabel("Mean per-run wall time (seconds)"); ax.set_xlabel("Method"); ax.legend(frameon=False, fontsize=7, ncol=2)
    finish(fig, "training_cost_components")
    notes = f"""# 图示说明（中文）

当前状态：{'**PROVISIONAL / numerical-noise：不作正式方法比较**' if meta.get('analysis_status') != 'active' else ('**PARTIAL，禁止作为完整正式结果引用**' if partial else '**active 注册表全部完成**')}；{meta['completed_endpoints']}/{meta['expected_endpoints']} 个端点。数据来源仅为明确指定注册表，未混入 pilot。当前端点为 {meta['split']} / {meta['checkpoint']}。

检查点曝光口径：{meta['checkpoint_exposure_policy']}。`target_prediction_tokens`/`replay_prediction_tokens` 是被评估检查点的实际曝光，`full_training_*` 是完整训练支出；best 不能被当作全部经历了最终目标预算的模型。

- `gain_forgetting_scatter`：分别展示 C4 与 WikiText-103。横轴目标 NLL 降低量越大越好，纵轴旧域 NLL 增量越小越好；小点为独立训练 seed，大点与横纵误差棒为 seed 均值 ± 样本标准差（ddof=1）。n<2 时不绘制 SD，不能解释为 SD=0。主图使用 b=.05，B0 使用 b=0。
- `budget_curves`：B4 与 M 在 b=.01/.05/.10 下的目标收益和两个旧域遗忘量，点与误差棒为实际已完成 seeds 的均值 ± SD；partial 图不能把完成较早的子集与其他方法作最终比较。详见 CSV 的 n/expected_n。
- `validation_trajectories`：细线为 seed，粗线为当前同曝光验证结果均值；不同时间点可能具有不同 n，原始表保留曝光和完成状态，不对缺失 seed 插值。该图始终使用验证数据，与测试端点区分。
- `training_cost_components`：每个正式运行的 setup、训练、验证、checkpoint 父计时的平均分项。教师和 KL 子计时已包含其中，不再次相加。共享生成/联合特征提取/选库开销另见 `shared_preparation_costs.csv`，每 seed 仅记一次。发生恢复时，逻辑 counters 可能不包含失败并回滚的工作，完整成本需外部 wall time。

`paired_statistics.csv` 的差值为同 seed 的 A−B 后再求均值和 SD，不是两个独立组均值相减再当作成对不确定性。文档 bootstrap CI（如启用测试 JSON 接口）条件于已训练 checkpoints，**不能替代训练 seed 的 SD**；每次共同抽取整篇文档及所有块，同时输出 token 加权和等文档平均。

`identical_bank_controls.csv` 单列 D−B4。若两者实际 tensor hash 相同，则去重未改变任何库内容；两次训练的差异是数值一致性诊断，不能解释成去重干预收益，也不能把它当作新的独立数据实验。

`matched_validation_loss.csv` 只描述共同目标验证 loss 范围内的首次下降穿越点，使用分段线性插值并保存两端实际检查点；非单调轨迹明确标记、禁止外推。这些是验证曲线诊断，**不是实际检查点、等预算比较或测试选模结果**，多个插值点也不增加训练重复数。
"""
    (output / "figure_notes_zh.md").write_text(notes)
