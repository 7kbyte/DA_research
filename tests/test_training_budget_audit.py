import importlib.util
from pathlib import Path

import pytest
import torch
from replay_study.train import TokenStream, load_training_data

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_training_budget.py'
spec = importlib.util.spec_from_file_location('training_budget_audit', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('ratio,prediction,raw,padded,calls', [
    (0., 0, 0, 0, 0),
    (.01, 100_000, 175_257, 238_899, 306),
    (.05, 500_000, 563_443, 626_176, 306),
    (.1, 1_000_000, 1_064_421, 1_126_912, 306),
])
def test_independent_reconstruction_matches_observed_frozen_primary_counts(ratio, prediction, raw, padded, calls):
    cfg = {'target_prediction_tokens': 10_000_000, 'microbatch_size': 16,
           'gradient_accumulation_steps': 4, 'replay_ratio': ratio}
    result = module.schedule_counts(cfg)
    assert [result['replay'][field] for field in ['prediction_tokens', 'raw_input_tokens',
            'padded_input_tokens', 'student_forward_calls_derived']] == [prediction, raw, padded, calls]
    assert [result['target'][field] for field in ['prediction_tokens', 'raw_input_tokens',
            'padded_input_tokens', 'student_forward_calls_derived']] == [10_000_000, 10_019_570, 10_019_840, 1224]


def test_scalar_reconstruction_matches_real_frozen_stream_across_partial_rows_and_padding(tmp_path):
    path = tmp_path / 'full_rows.pt'
    torch.save({'input_ids': torch.arange(27).reshape(3, 9)}, path)
    stream = TokenStream(load_training_data(path), seed=11)
    takes = [3, 7, 1, 9, 2, 5]
    batches = [batch for count in takes for batch in stream.take(count, 2)]
    expected = module.uniform_stream_counts(takes, batch_size=2, sequence_length=9)
    for field in ['prediction_tokens', 'raw_input_tokens', 'padded_input_tokens']:
        assert expected[field] == sum(batch[field] for batch in batches)
    assert expected['student_forward_calls_derived'] == len(batches)
    assert expected['final_offset'] == stream.offset
    # The second request retains the consumed prefix as context but masks its loss.
    continued = batches[1]
    assert continued['prediction_mask'][0, :3].sum() == 0
    assert continued['prediction_mask'][0, 3:].sum() > 0
    assert bool(continued['attention_mask'][0, :4].all())

