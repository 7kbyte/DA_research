"""Printable A-only sensitivity figure, using completed authorized CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completion-receipt', type=Path, required=True)
    p.add_argument('--completion-receipt-sha256', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    if sha(args.completion_receipt) != args.completion_receipt_sha256:
        raise ValueError('A completion receipt SHA mismatch')
    receipt = json.loads(args.completion_receipt.read_text())
    if receipt['status'] != 'complete' or receipt['scope'] != 'ONLY authorized A-NLL':
        raise ValueError('Authorized complete A analysis is required')
    for path, expected in receipt['files_sha256'].items():
        if sha(path) != expected:
            raise ValueError('A completion source changed: ' + path)
    root = Path('results/supplemental_analysis_v2/supplement_A_nll').resolve()
    sources = {str(args.completion_receipt.resolve()): sha(args.completion_receipt)}

    def load(name):
        path = root / name
        sources[str(path)] = sha(path)
        with path.open() as handle:
            return list(csv.DictReader(handle))

    points = load('final/endpoints.csv')
    statistics = load('final/endpoint_statistics.csv')
    selected = load('validation_selected_policy/selected_endpoints.csv')
    methods = ('B1', 'B2', 'B3', 'B4', 'M', 'C')
    weights = (.01, .05, .1, .5, 1., 5., 10.)
    expected_selection = {'B1': .05, 'B2': 1., 'B3': .1, 'B4': 5., 'M': 5., 'C': 1.}
    metrics = ('g_new', 'f_old_c4', 'f_old_wikitext')
    if len(points) != 126 or len(selected) != 18:
        raise ValueError('Incomplete A grid or selected policy')
    for method in methods:
        for weight in weights:
            cell = [x for x in points if x['group'] == method and float(x['weight']) == weight]
            if {int(x['training_seed']) for x in cell} != {11, 22, 33} or len(cell) != 3:
                raise ValueError('Incomplete six-method seven-lambda three-seed grid')
        chosen = [x for x in selected if x['group'] == method]
        if len(chosen) != 3 or {float(x['weight']) for x in chosen} != {expected_selection[method]}:
            raise ValueError('Frozen validation selection changed')
    if {x['validation_selection_sha256'] for x in selected} != {'9548085fc18278f69fad584b67485bceb5193a18b10c78f516f6b09b4fa4e477'}:
        raise ValueError('Unexpected validation selection source')
    stats = {(x['group'], float(x['weight']), x['metric']): x for x in statistics}
    out = args.output_root.resolve()
    if out.exists():
        raise FileExistsError('Figure output is immutable')
    out.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b')
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True)
    labels = ('G_new (higher is better)', 'F_C4 (lower is better)', 'F_Wiki (lower is better)')
    for axis, metric, label in zip(axes, metrics, labels):
        for method, color in zip(methods, colors):
            means = [float(stats[method, weight, metric]['mean']) for weight in weights]
            sds = [float(stats[method, weight, metric]['std_ddof1']) for weight in weights]
            axis.errorbar(weights, means, yerr=sds, color=color, linewidth=1, elinewidth=.7, capsize=1.8, alpha=.95, zorder=2)
            for seed, marker in ((11, 'o'), (22, 's'), (33, '^')):
                cell = sorted([x for x in points if x['group'] == method and int(x['training_seed']) == seed], key=lambda x: float(x['weight']))
                axis.scatter([float(x['weight']) for x in cell], [float(x[metric]) for x in cell], marker=marker, color=color, s=10, alpha=.58, linewidths=0, zorder=3)
            chosen_weight = expected_selection[method]
            mean = float(stats[method, chosen_weight, metric]['mean'])
            axis.scatter([chosen_weight], [mean], marker='D', s=54, facecolors='none', edgecolors=color, linewidths=1.4, zorder=5)
        axis.axhline(0, color='0.3', linestyle='--', linewidth=.8, zorder=1)
        axis.set_xscale('log')
        axis.set_ylabel(label, fontsize=8.5)
        axis.tick_params(axis='both', labelsize=8)
        axis.grid(alpha=.16, linewidth=.6)
        axis.set_xlim(.008, 12.5)
    axes[-1].set_xticks(weights, ['.01', '.05', '.1', '.5', '1', '5', '10'])
    axes[-1].set_xlabel('Replay loss weight lambda (log scale)', fontsize=9)
    handles = [Line2D([], [], color=color, label=method + f' (val lambda={expected_selection[method]:g})') for method, color in zip(methods, colors)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .96), ncol=3, frameon=False, fontsize=8, columnspacing=1.2, handlelength=1.5)
    fig.suptitle('Supplement A: complete sensitivity grid; 3 seeds, mean +/- sample SD', fontsize=10, y=.995)
    fig.text(.53, .865, 'Open diamonds: frozen validation choices.  Seeds: 11 circle / 22 square / 33 triangle.', ha='center', fontsize=7.5)
    fig.subplots_adjust(left=.16, right=.98, top=.835, bottom=.085, hspace=.18)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / ('A_complete_sensitivity_selected_validation.' + ext), dpi=200)
    plt.close(fig)
    for name, rows in (('all_126_seed_points.csv', points), ('all_42_candidate_mean_sample_sd.csv', [x for x in statistics if x['metric'] in metrics]), ('validation_selected_18_points.csv', selected)):
        with (out / name).open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    proof = {'scope': 'ONLY_authorized_A_NLL_display', 'source_sha256': sources, 'plotter_sha256': sha(__file__), 'output_sha256': {str(x): sha(x) for x in out.iterdir()}, 'dimensions_inches': [7, 6.5], 'logical_candidates': 42, 'trained_checkpoints': 126, 'seed_points_across_three_metrics': 378, 'validation_markers': 18, 'selection_recomputed': False, 'sensitivity_is_descriptive_not_test_selection': True, 'other_scores_remain_sealed': True}
    path = out / 'figure_provenance.json'
    path.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output_root': str(out), 'provenance_sha256': sha(path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
