#!/usr/bin/env python3
"""Render readable report-sized primary validation trajectories from audited CSV."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path,
                        default=Path('results/analysis/formal_deterministic_validation'))
    parser.add_argument('--output', type=Path, default=Path('results/report_figures'))
    args = parser.parse_args()
    source = args.analysis_root / 'trajectories.csv'
    provenance = args.analysis_root / 'analysis_manifest.json'
    meta = json.loads(provenance.read_text())
    # Explicitly restricted to the completed primary validation table.
    if (not meta.get('complete') or meta.get('split') != 'validation' or meta.get('analysis_status') != 'active'
            or meta.get('completed_endpoints') != 36 or meta.get('expected_endpoints') != 36
            or meta.get('model_revision') != '9879c9b5f8bea9051dcb0e68dff21493d67e9d4f'):
        raise ValueError('Completed active primary validation analysis required')
    frame = pd.read_csv(source)
    if frame.target_prediction_tokens.max() != 10_000_000:
        raise ValueError('This primary figure requires the frozen 10M horizon')
    frame = frame[(frame.group == 'B0') | np.isclose(frame.ratio, .05)].copy()
    metrics = ['g_new', 'f_old_c4', 'f_old_wikitext']
    key = ['seed', 'target_prediction_tokens']
    left = frame[frame.group == 'B4'].sort_values(key)
    right = frame[frame.group == 'D'].sort_values(key)
    if not np.array_equal(left[key + metrics].to_numpy(), right[key + metrics].to_numpy()):
        raise ValueError('D cannot be hidden as an identical B4 curve')
    colors = {'B0':'#666666', 'B1':'#E69F00', 'B2':'#56B4E9', 'B3':'#009E73',
              'B4':'#0072B2', 'M':'#D55E00', 'C':'#CC79A7'}
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'svg.fonttype':'path'})
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.8), sharex=True)
    for ax, metric, label in zip(axes, metrics,
            ['Math adaptation gain (higher is better)', 'C4 forgetting (lower is better)',
             'WikiText-103 forgetting (lower is better)']):
        for group, color in colors.items():
            current = frame[frame.group == group]
            if sorted(current.seed.unique()) != [11, 22, 33]:
                raise ValueError('All three primary seeds required for each displayed group')
            for _, points in current.groupby('seed'):
                points = points.sort_values('target_prediction_tokens')
                x = np.r_[0, points.target_prediction_tokens.to_numpy() / 1e6]
                y = np.r_[0, points[metric].to_numpy()]
                ax.plot(x, y, color=color, alpha=.18, lw=.7)
            mean = current.groupby('target_prediction_tokens')[metric].mean().sort_index()
            ax.plot(np.r_[0, mean.index.to_numpy()/1e6], np.r_[0, mean.to_numpy()],
                    color=color, label=group, lw=1.9, linestyle='--' if group in ['B1','B3'] else '-')
        ax.set_title(label, loc='left', fontsize=10.5)
        ax.set_ylabel('nats / token')
        ax.grid(alpha=.16)
        ax.axhline(0, color='.65', lw=.6)
        ax.set_xlim(0, 10)
    axes[0].legend(ncol=7, loc='lower left', bbox_to_anchor=(-.02,1.16), frameon=False,
                   handlelength=1.8, columnspacing=1.1, fontsize=9)
    axes[-1].set_xlabel('Target prediction-token exposure (million)')
    fig.suptitle('Primary validation trajectories', y=.995, fontsize=12)
    fig.text(.12,.012,'Thin lines: three seeds; thick lines: mean. D = B4 exactly.\n'
             'Zero-exposure anchor follows the metric definition; connecting lines are guides.', fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.965), h_pad=1.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ['png','pdf','svg']:
        fig.savefig(args.output / ('primary_validation_trajectories.'+extension), dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    record = {'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [source,provenance,Path(__file__)]},
              'split':'validation','primary_seeds':[11,22,33], 'main_replay_ratio':.05,
              'B0_replay_ratio':0, 'D_omitted_only_after_exact_curve_comparison':True,
              'zero_anchor':'G=F=0 at unchanged initial model; not an extra trained checkpoint',
              'line_interpolation_is_not_a_measured_intermediate_checkpoint':True}
    (args.output/'primary_validation_trajectories.sources.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'displayed_groups':list(colors)}))


if __name__ == '__main__':
    main()
