#!/usr/bin/env python3
"""Post-test descriptive decomposition of the two predeclared target weightings."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replay_study.analysis import document_totals, write_csv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol_path = ROOT / 'docs/primary_target_weighting_decomposition_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    sources = {str(protocol_path): digest(protocol_path), str(Path(__file__)): digest(__file__)}
    source_analysis = ROOT / 'src/replay_study/analysis.py'
    sources[str(source_analysis)] = digest(source_analysis)
    receipt_path = ROOT / 'docs/primary_nll_first_test_read.json'
    receipt = json.loads(receipt_path.read_text())
    manifest_path = ROOT / protocol['manifest']
    assert digest(manifest_path) == protocol['manifest_sha256'] == receipt['manifest_sha256']
    assert receipt['all_67_jobs_completed_and_integrity_verified'] is True
    sources[str(receipt_path)] = digest(receipt_path)
    sources[str(manifest_path)] = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    authorized = {}
    endpoint_reference = {}
    for endpoint in ('final', 'best'):
        directory = ROOT / 'results/analysis/primary_nll_v2' / endpoint
        analysis_path = directory / 'analysis_manifest.json'
        analysis = json.loads(analysis_path.read_text())
        assert analysis['complete'] and analysis['completed_conditions'] == 36
        assert analysis['identity']['role'] == 'primary'
        for path, expected in analysis['sources_sha256'].items():
            assert digest(path) == expected, path
            authorized[str(Path(path).resolve())] = expected
        sources[str(analysis_path)] = digest(analysis_path)
        endpoints_path = directory / 'endpoints.csv'
        sources[str(endpoints_path)] = digest(endpoints_path)
        for row in csv.DictReader(endpoints_path.open()):
            endpoint_reference[(endpoint, row['run_id'])] = row
    cache = {}

    def target(path):
        path = Path(path).resolve()
        if path not in cache:
            assert str(path) in authorized
            assert digest(path) == authorized[str(path)]
            sources[str(path)] = authorized[str(path)]
            value = json.loads(path.read_text())['corpora']['target']
            docs = document_totals(value)
            assert len(docs) == 466
            assert all(n % 511 == 0 and 1 <= n // 511 <= 4 for _, n in docs.values())
            cache[path] = docs
        return cache[path]

    rows, bins, document_rows = [], [], []
    for record in manifest['records']:
        assert record['role'] == 'primary' and record['kind'] in {'final', 'best'}
        identity = dict(endpoint=record['kind'], run_id=record['run_id'], group=record['group'],
                        training_seed=record['training_seed'], ratio=record['ratio'], weight=record['weight'],
                        target_prediction_tokens=record['target_prediction_tokens_at_checkpoint'])
        baseline = target(manifest['baseline_groups'][record['baseline_group']]['initial_output'])
        adapted = target(record['output'])
        assert baseline.keys() == adapted.keys()
        values = []
        for doc_id, (adapted_sum, count) in adapted.items():
            initial_sum, initial_count = baseline[doc_id]
            assert initial_count == count
            gain = (initial_sum - adapted_sum) / count
            values.append((count, gain))
            document_rows.append(dict(**identity, doc_id=doc_id, retained_chunks=count // 511,
                                      retained_prediction_tokens=count, target_gain=gain))
        mean_count = statistics.mean(n for n, _ in values)
        equal_gain = statistics.mean(g for _, g in values)
        token_gain = math.fsum(n * g for n, g in values) / sum(n for n, _ in values)
        covariance = math.fsum((n - mean_count) * (g - equal_gain) for n, g in values) / len(values)
        gap = token_gain - equal_gain
        residual = gap - covariance / mean_count
        assert abs(residual) < 1e-12
        reference = endpoint_reference[(record['kind'], record['run_id'])]
        assert math.isclose(token_gain, float(reference['g_new']), abs_tol=1e-12)
        assert math.isclose(equal_gain, float(reference['g_new_equal_document']), abs_tol=1e-12)
        rows.append(dict(**identity, documents=len(values), prediction_tokens=sum(n for n, _ in values),
                         token_weighted_gain=token_gain, equal_document_gain=equal_gain,
                         token_minus_document_gain=gap, population_covariance_tokens_gain=covariance,
                         mean_retained_prediction_tokens=mean_count, identity_residual=residual))
        for chunks in (1, 2, 3, 4):
            gains = [g for n, g in values if n == chunks * 511]
            assert gains
            bins.append(dict(**identity, retained_chunks=chunks, documents=len(gains),
                             mean_document_gain=statistics.mean(gains),
                             document_sd_ddof1=statistics.stdev(gains)))
    assert len(rows) == 72
    output = ROOT / 'results/analysis/primary_target_weighting_posttest'
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'endpoint_decomposition.csv', rows)
    write_csv(output / 'retained_chunk_groups.csv', bins)
    write_csv(output / 'all_document_gains.csv', document_rows)
    plot(bins, output)
    for path, expected in sources.items():
        assert digest(path) == expected, 'input changed: ' + path
    metadata = dict(generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    status='complete_post_test_descriptive_decomposition', protocol=protocol,
                    endpoints=len(rows), documents_per_endpoint=466,
                    max_identity_residual=max(abs(r['identity_residual']) for r in rows),
                    inference_jobs_added=0, training_jobs_added=0, sources_sha256=sources,
                    uncertainty='Within-bin document SD is descriptive, not a model CI; plot points are three pipeline seeds',
                    output_sha256={p.name: digest(p) for p in output.iterdir() if p.name != 'manifest.json'})
    (output / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: metadata[k] for k in ('status', 'endpoints', 'max_identity_residual')}))


def plot(bins, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    groups = ('B0', 'B1', 'B2', 'B3', 'B4', 'M', 'C')
    colors = dict(zip(groups, ('#252525', '#e69f00', '#009e73', '#56b4e9', '#0072b2', '#d55e00', '#cc79a7')))
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True, sharey=True, layout='constrained')
    for ax, endpoint in zip(axes, ('final', 'best')):
        all_rows = [r for r in bins if r['endpoint'] == endpoint]
        for seed in (11, 22, 33):
            for chunks in (1, 2, 3, 4):
                null = {r['group']: r['mean_document_gain'] for r in all_rows
                        if r['group'] in {'D', 'B4'} and r['training_seed'] == seed and r['ratio'] == .05
                        and r['retained_chunks'] == chunks}
                assert null['D'] == null['B4']
        for group in groups:
            selected = [r for r in all_rows if r['group'] == group and r['ratio'] == (0 if group == 'B0' else .05)]
            means = []
            for chunks in (1, 2, 3, 4):
                subset = [r['mean_document_gain'] for r in selected if r['retained_chunks'] == chunks]
                assert len(subset) == 3
                means.append(statistics.mean(subset))
                ax.scatter([chunks] * 3, subset, s=12, color=colors[group], alpha=.45)
            ax.plot(range(1, 5), means, color=colors[group], marker='o', ms=4, label=group)
        ax.axhline(0, color='.5', lw=.7, ls='--')
        ax.set_ylabel('Math adaptation gain (nats/token)')
        ax.set_title('Final: fixed 10M target exposure' if endpoint == 'final'
                     else 'Best: selected by target validation; exposure varies')
        ax.grid(alpha=.18)
    counts = {r['retained_chunks']: r['documents'] for r in bins}
    axes[-1].set_xticks(range(1, 5), [f'{k}\n{counts[k]} documents' for k in range(1, 5)])
    axes[-1].set_xlabel('Retained evaluation chunks per document (511 prediction tokens per chunk)')
    axes[0].legend(ncol=7, fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.29))
    fig.suptitle('Post-test descriptive weighting decomposition\nD equals B4 exactly and is omitted; points: seeds, lines: means', fontsize=10)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'retained_chunks_target_gain.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
