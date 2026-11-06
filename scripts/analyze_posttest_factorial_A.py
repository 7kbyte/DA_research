"""Post-test exploratory A-only source x objective contrast; no model inference.

Consumes only already-authorized derived endpoint and paired-effect CSVs. It
does not rerun primary/A statistics or use any selected-policy comparison.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_SHA256 = "18de2ca4e8ff45feebb69e9499ba3bde17e01f58a086ac831dcda183a4dd4187"
PAIRS_SHA256 = "6b91081526b28a8b01e6de9b60ba9a0c752bd019995afb0488079f507d881f6c"
METRICS = (
    "g_new", "f_old_c4", "f_old_wikitext", "g_new_equal_document",
    "f_old_c4_equal_document", "f_old_wikitext_equal_document",
)
WEIGHTS = (.01, .05, .1, .5, 1., 5., 10.)
SEEDS = (11, 22, 33)
GROUPS = ("B1", "B2", "B3", "B4")
COMPARISONS = ("B4-B3", "B2-B1", "B4-B2", "B3-B1")
FORMULA = "(B4-B3)-(B2-B1)"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError("Nonfinite source value: " + key)
    return value


def indexed(rows, field, choices):
    result = {}
    for row in rows:
        if row[field] not in choices:
            continue
        key = (row[field], finite(row, "weight"), int(row["training_seed"]))
        if key in result:
            raise ValueError("Duplicate source cell: " + str(key))
        result[key] = row
    expected = {(c, w, s) for c in choices for w in WEIGHTS for s in SEEDS}
    if set(result) != expected:
        raise ValueError("Incomplete or unexpected factorial grid")
    return result


def interactions(endpoints, pairs):
    """Return long individual contrasts, 42 summaries, and algebra checks.

    The two differences are paired within a pipeline seed. Each lambda has
    exactly three observations; the seven lambda settings are not replicates.
    """
    cells = indexed(endpoints, "group", GROUPS)
    effects = indexed(pairs, "comparison", COMPARISONS)
    baseline_groups = set()
    for row in cells.values():
        if (row["role"], row["phase_name"], row["model_tag"],
                row["data_family"], row["checkpoint"]) != (
                    "supplement_A", "supplement_A", "pythia-410m", "main_v3", "final"):
            raise ValueError("Source identity is outside the declared A final scope")
        for key, expected in (("target_prediction_tokens", 10_000_000),
                              ("replay_prediction_tokens", 500_000),
                              ("full_training_target_prediction_tokens", 10_000_000),
                              ("full_training_replay_prediction_tokens", 500_000)):
            if int(row[key]) != expected:
                raise ValueError("Unequal or incorrect actual exposure: " + key)
        if finite(row, "ratio") != .05 or finite(row, "learning_rate") != 3e-5:
            raise ValueError("Frozen ratio or learning rate differs")
        baseline_groups.add(row["baseline_group"])
    if len(baseline_groups) != 1:
        raise ValueError("Different baseline/data/policy identity in factorial cells")
    individual, checks = [], []
    for weight in WEIGHTS:
        for seed in SEEDS:
            four = {g: cells[g, weight, seed] for g in GROUPS}
            for name in COMPARISONS:
                left, right = name.split("-")
                row = effects[name, weight, seed]
                if (row["run_a"], row["run_b"]) != (four[left]["run_id"], four[right]["run_id"]):
                    raise ValueError("Paired-effect run identity differs")
                if (int(row["training_seed_a"]), int(row["training_seed_b"])) != (seed, seed):
                    raise ValueError("Paired effect mixes seeds")
                if (int(row["target_tokens_a"]), int(row["target_tokens_b"])) != (10_000_000, 10_000_000):
                    raise ValueError("Paired effect mixes checkpoint exposure")
                if row["equal_checkpoint_exposure"] != "True":
                    raise ValueError("Paired effect lacks equal exposure")
            for metric in METRICS:
                values = {g: finite(row, metric) for g, row in four.items()}
                direct_effects = {name: values[name[:2]] - values[name[-2:]] for name in COMPARISONS}
                existing = {name: finite(effects[name, weight, seed], metric) for name in COMPARISONS}
                direct = direct_effects["B4-B3"] - direct_effects["B2-B1"]
                objective_reconstruction = existing["B4-B3"] - existing["B2-B1"]
                source_reconstruction = existing["B4-B2"] - existing["B3-B1"]
                errors = [abs(existing[n] - direct_effects[n]) for n in COMPARISONS]
                errors += [abs(direct - objective_reconstruction), abs(direct - source_reconstruction)]
                if max(errors) > 1e-12:
                    raise ValueError("Existing paired algebra differs at " + str((weight, seed, metric)))
                result = {
                    "analysis_status": "post_test_exploratory_not_preregistered",
                    "formula": FORMULA, "weight": weight, "training_seed": seed,
                    "metric": metric, "weighting": "equal_document" if metric.endswith("equal_document") else "prediction_token",
                    "contrast_direction": "positive means larger generated-versus-C4 KL-CE gain contrast" if metric.startswith("g_new") else "negative means smaller generated-versus-C4 KL-CE forgetting contrast",
                    "B1": values["B1"], "B2": values["B2"], "B3": values["B3"], "B4": values["B4"],
                    "generated_KL_minus_CE": direct_effects["B4-B3"],
                    "c4_KL_minus_CE": direct_effects["B2-B1"], "interaction": direct,
                    "ratio": .05, "learning_rate": 3e-5, "checkpoint": "final",
                    "model_tag": "pythia-410m", "data_family": "main_v3",
                    "baseline_group": next(iter(baseline_groups)),
                }
                for group, row in four.items():
                    for key in ("record_id", "run_id", "actual_run_id", "inference_job_id",
                                "target_prediction_tokens", "replay_prediction_tokens",
                                "full_training_target_prediction_tokens", "full_training_replay_prediction_tokens",
                                "is_training_alias", "is_inference_alias"):
                        result[group + "_" + key] = row[key]
                individual.append(result)
                checks.append({
                    "weight": weight, "training_seed": seed, "metric": metric,
                    "direct_interaction": direct,
                    "from_existing_objective_pairs": objective_reconstruction,
                    "from_existing_source_pairs": source_reconstruction,
                    "max_abs_error_across_four_terms_and_two_identities": max(errors),
                    "checked_scalar_equalities": 6, "absolute_tolerance": 1e-12, "passed": True,
                })
    summaries = []
    for weight in WEIGHTS:
        for metric in METRICS:
            values = [r["interaction"] for r in individual if r["weight"] == weight and r["metric"] == metric]
            summaries.append({
                "weight": weight, "metric": metric, "mean": statistics.mean(values),
                "std_ddof1": statistics.stdev(values), "n": 3, "training_seeds": "11,22,33",
                "replicate_unit": "paired_pipeline_seed_within_fixed_lambda",
                "uncertainty_scope": "sample SD of three paired contrasts; not CI; settings share seeds/data/banks",
                "analysis_status": "post_test_exploratory_not_preregistered", "formula": FORMULA,
                "target_prediction_tokens_each_checkpoint": 10_000_000,
                "replay_prediction_tokens_each_checkpoint": 500_000,
            })
    return individual, summaries, checks


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(individual, summaries, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(3, 2, figsize=(7, 6.6), sharex=True)
    labels = ("I(G_new): higher contrast", "I(F_C4): lower contrast", "I(F_Wiki): lower contrast")
    colors = ("#377eb8", "#e41a1c", "#4daf4a")
    for column, suffix in enumerate(("", "_equal_document")):
        for row, base in enumerate(METRICS[:3]):
            metric = base + suffix
            axis = axes[row, column]
            stat = [r for r in summaries if r["metric"] == metric]
            axis.errorbar(WEIGHTS, [r["mean"] for r in stat], yerr=[r["std_ddof1"] for r in stat],
                          color="#202020", linewidth=1.1, capsize=2, marker="D", markersize=3, zorder=4)
            for seed, marker, color in zip(SEEDS, ("o", "s", "^"), colors):
                points = [r for r in individual if r["metric"] == metric and r["training_seed"] == seed]
                axis.scatter([r["weight"] for r in points], [r["interaction"] for r in points],
                             s=17, marker=marker, color=color, alpha=.85, zorder=5)
            axis.axhline(0, color=".4", linewidth=.8, linestyle="--")
            axis.set_xscale("log"); axis.set_xlim(.0075, 13.5)
            axis.set_ylabel(labels[row] + "\n(nats per token)", fontsize=8)
            axis.tick_params(labelsize=8); axis.grid(alpha=.15, linewidth=.5)
            if row == 0:
                axis.set_title(("Prediction-token weighting", "Equal-document weighting")[column], fontsize=9)
        axes[-1, column].set_xticks(WEIGHTS, [".01", ".05", ".1", ".5", "1", "5", "10"])
        axes[-1, column].set_xlabel("Fixed loss weight lambda (log)", fontsize=8)
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", markersize=4, label="Seed " + str(s))
               for s, m, c in zip(SEEDS, ("o", "s", "^"), colors)]
    handles.append(Line2D([], [], color="#202020", marker="D", markersize=3, label="Mean +/- sample SD"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .925), ncol=4, frameon=False, fontsize=7.5)
    fig.suptitle("Post-test exploratory interaction: (B4-B3)-(B2-B1)", fontsize=10, y=.99)
    fig.text(.51, .945, "All seven settings shown; shared seeds/data/banks across lambda. No test-based choice.", ha="center", fontsize=7.5)
    fig.text(.51, .017, "Signs concern the interaction contrast, not overall superiority or a mechanism test. Error bars are not CIs.", ha="center", fontsize=7)
    fig.subplots_adjust(left=.12, right=.985, top=.835, bottom=.10, wspace=.39, hspace=.30)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / ("posttest_factorial_interaction_A." + extension), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("docs/posttest_factorial_interaction_A_protocol.json"))
    args = parser.parse_args()
    started = time.monotonic()
    if sha(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("Post-test declaration changed")
    declaration = json.loads(args.protocol.read_text())
    if declaration["metrics"] != list(METRICS) or declaration["interaction_formula"] != FORMULA + ", paired within the same seed and lambda":
        raise ValueError("Declared formula/metrics differ")
    source = Path(declaration["source"]["path"])
    paired_source = source.with_name("paired_effects.csv")
    for path, expected in ((source, declaration["source"]["sha256"]), (paired_source, PAIRS_SHA256)):
        if sha(path) != expected:
            raise ValueError("Authorized derived source changed: " + str(path))
    with source.open() as handle:
        endpoints = list(csv.DictReader(handle))
    with paired_source.open() as handle:
        pairs = list(csv.DictReader(handle))
    if len(endpoints) != 126:
        raise ValueError("Complete A126 derived source required")
    individual, summaries, checks = interactions(endpoints, pairs)
    out = Path(declaration["output_root"])
    out.mkdir(parents=True, exist_ok=False)
    selected_sources = [r for r in endpoints if r["group"] in GROUPS]
    selected_sources.sort(key=lambda r: (float(r["weight"]), int(r["training_seed"]), r["group"]))
    write_csv(out / "source_endpoints_84.csv", selected_sources)
    write_csv(out / "interaction_individual_126.csv", individual)
    write_csv(out / "interaction_mean_sample_sd_42.csv", summaries)
    write_csv(out / "algebra_consistency_126.csv", checks)
    plot(individual, summaries, out)
    elapsed = time.monotonic() - started
    manifest = {
        "status": "completed_post_test_exploratory_not_preregistered", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(p.resolve()): sha(p) for p in (args.protocol, source, paired_source)},
        "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "output_sha256": {str(p.resolve()): sha(p) for p in sorted(out.iterdir())},
        "formula": FORMULA, "settings": 7, "paired_seeds_per_setting": 3, "source_checkpoints": 84,
        "individual_metric_contrasts": len(individual), "summary_rows": len(summaries),
        "algebra_scalar_checks": sum(r["checked_scalar_equalities"] for r in checks),
        "max_abs_algebra_error": max(r["max_abs_error_across_four_terms_and_two_identities"] for r in checks),
        "sample_sd_ddof": 1, "confidence_intervals_or_bootstrap_or_p_values_added": False,
        "test_based_lambda_selection_performed": False, "across_lambda_settings_independent": False,
        "original_primary_A_or_selected_statistics_recomputed_or_modified": False,
        "new_training_or_model_forward_calls": 0, "B_C_or_behavior_scores_read": False,
        "measured_CPU_analysis_and_plot_wall_seconds": elapsed, "timing_scope": "this new arithmetic/plot operation, not old inference or parent A analysis",
        "figure_dimensions_inches": [7, 6.6],
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_root": str(out.resolve()), "manifest_sha256": sha(out / "analysis_manifest.json"), "analysis_wall_seconds": elapsed}))


if __name__ == "__main__":
    main()
