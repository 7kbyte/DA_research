import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from replay_study.train import TrainConfig, TokenStream, load_training_data, replay_tokens_due, train


class TinyCausalModel(torch.nn.Module):
    def __init__(self, vocab=17, hidden=9):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.projection = torch.nn.Linear(hidden, vocab)
        self.config = SimpleNamespace(use_cache=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        embedded = self.embedding(input_ids)
        if attention_mask is not None:
            embedded = embedded * attention_mask.unsqueeze(-1)
        hidden = embedded.cumsum(1) / torch.arange(1, input_ids.shape[1] + 1, device=input_ids.device)[None, :, None]
        hidden = torch.nn.functional.dropout(hidden, p=0.1, training=self.training)
        return SimpleNamespace(logits=self.projection(hidden))


def make_data(path, rows=7, length=10):
    generator = torch.Generator().manual_seed(5)
    torch.save({"input_ids": torch.randint(17, (rows, length), generator=generator)}, path)
    return path


def config(tmp_path, output, **kwargs):
    target = tmp_path / "target.pt"
    replay = tmp_path / "replay.pt"
    if not target.exists():
        make_data(target)
        make_data(replay, rows=4)
    return TrainConfig(model_name_or_path="injected-tiny", target_path=str(target),
                       replay_path=str(replay), output_dir=str(tmp_path / output),
                       target_prediction_tokens=113, target_tokens_per_step=20,
                       replay_objective="kl", replay_ratio=.05, replay_weight=.5,
                       microbatch_size=2, gradient_accumulation_steps=1,
                       precision="fp32", device="cpu", fused_adamw=False,
                       learning_rate=1e-3, save_every_steps=0, eval_every_steps=0,
                       evaluate_at_end=False, **kwargs)


def test_low_replay_ratios_have_no_per_step_rounding_bias():
    for ratio, numerator in ((.01, 1), (.05, 5), (.1, 10)):
        actual = 0
        for cumulative in range(1, 10004, 37):
            due = replay_tokens_due(cumulative, ratio, actual)
            assert due >= 0
            actual += due
            assert actual == cumulative * numerator // 100
        assert replay_tokens_due(10_000_000, ratio) == 100_000 * numerator


def test_partial_chunk_stream_restores_and_does_not_drop_tail(tmp_path):
    data = load_training_data(make_data(tmp_path / "data.pt", rows=3, length=8))
    stream = TokenStream(data, seed=11)
    first = list(stream.take(5, 2))
    assert sum(b["prediction_tokens"] for b in first) == 5
    saved = stream.state_dict()
    continued = list(stream.take(19, 2))
    restored = TokenStream(data, seed=11)
    restored.load_state_dict(saved)
    repeated = list(restored.take(19, 2))
    assert sum(b["prediction_tokens"] for b in repeated) == 19
    for a, b in zip(continued, repeated):
        for key in ("input_ids", "attention_mask", "prediction_mask"):
            assert torch.equal(a[key], b[key])
    assert stream.state_dict() == restored.state_dict()
    assert stream.exposure == 24
    assert stream.epoch == 1
    assert stream.offset == 3
    assert stream.unique_prediction_tokens_exposed() == 21


def test_resume_is_identical_to_uninterrupted_training_and_token_budgets(tmp_path):
    torch.manual_seed(9)
    initial = TinyCausalModel()
    teacher = copy.deepcopy(initial)
    uninterrupted_model = copy.deepcopy(initial)
    full = train(config(tmp_path, "full"), uninterrupted_model, copy.deepcopy(teacher))
    partial_config = config(tmp_path, "resumed", max_steps=2)
    partial = train(partial_config, copy.deepcopy(initial), copy.deepcopy(teacher))
    assert partial["status"] == "max_steps"
    # Simulate a crash leaving events newer than the durable checkpoint.
    with (Path(partial_config.output_dir) / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": "train_step", "step": 3, "not_durable": True}) + "\n")
    resumed_model = copy.deepcopy(initial)
    resumed_config = dataclasses.replace(partial_config, max_steps=None, resume_from=partial["checkpoint_path"])
    resumed = train(resumed_config, resumed_model, copy.deepcopy(teacher))
    assert resumed["status"] == "completed"
    assert 0 < resumed["counters"]["training_teacher_forward_seconds"] <= resumed["counters"]["training_seconds"]
    assert resumed["timing"]["training_teacher_forward_seconds_complete"]
    assert resumed["timing"]["training_teacher_forward_seconds_nested_in"] == "training_seconds"
    assert resumed["counters"]["target_prediction_tokens"] == 113
    assert resumed["counters"]["replay_prediction_tokens"] == 5
    assert resumed["target_unique_prediction_tokens_exposed"] == 63
    assert resumed["replay_unique_prediction_tokens_exposed"] == 5
    assert resumed["coverage_accounting"]["target"]["repeated_prediction_token_exposures"] == 50
    assert resumed["target_stream"] == full["target_stream"]
    assert resumed["replay_stream"] == full["replay_stream"]
    for key, value in uninterrupted_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[key], atol=0, rtol=0)
    steps = [json.loads(line) for line in (Path(resumed_config.output_dir) / "events.jsonl").read_text().splitlines()
             if json.loads(line)["event"] == "train_step"]
    assert len(steps) == 6
    assert list(Path(resumed_config.output_dir).glob("events.pre_resume_*.jsonl"))
    assert sum(x["step_counts"]["target_prediction_tokens"] for x in steps) == 113
    assert sum(x["step_counts"]["replay_prediction_tokens"] for x in steps) == 5


def test_methods_share_target_stream_and_lr_and_validation_callback(tmp_path):
    torch.manual_seed(19)
    initial = TinyCausalModel()
    calls = []
    def validate(model, state):
        assert not model.training
        assert not torch.is_grad_enabled()
        calls.append(state["target_prediction_tokens"])
        return {"target": {"nll": 2.0, "teacher_forward_seconds": .125, "kl_arithmetic_seconds": .025}}
    replay_config = config(tmp_path, "replay", max_steps=2)
    replay_config.eval_every_steps = 1
    result = train(replay_config, copy.deepcopy(initial), copy.deepcopy(initial), validate)
    baseline_config = dataclasses.replace(replay_config, output_dir=str(tmp_path / "baseline"),
                                          replay_path=None, replay_objective="none", replay_ratio=0.)
    baseline = train(baseline_config, copy.deepcopy(initial))
    assert result["target_stream"] == baseline["target_stream"]
    assert baseline["counters"]["training_teacher_forward_seconds"] == 0.0
    assert calls == [20, 40]
    assert len(result["validation_trajectory"]) == 2
    assert result["counters"]["validation_teacher_forward_seconds"] == .25
    assert result["counters"]["validation_kl_arithmetic_seconds"] == .05
    assert result["best_target_validation"]["step"] == 1
    best = torch.load(result["best_target_validation"]["checkpoint_path"], weights_only=False)
    final = torch.load(result["checkpoint_path"], weights_only=False)
    assert best["checkpoint_kind"] == "model_only_best_target_validation"
    assert "optimizer_state" not in best
    assert best["counters"]["step"] == 1
    assert final["counters"]["step"] == 2
    def learning_rates(directory):
        return [json.loads(x)["learning_rate"] for x in (Path(directory) / "events.jsonl").read_text().splitlines()
                if json.loads(x)["event"] == "train_step"]
    assert learning_rates(replay_config.output_dir) == learning_rates(baseline_config.output_dir)


def test_expired_deadline_saves_resumable_checkpoint_without_exposure(tmp_path):
    torch.manual_seed(29)
    initial = TinyCausalModel()
    deadline_config = config(tmp_path, "deadline", deadline_utc="2020-01-01T00:00:00Z")
    stopped = train(deadline_config, copy.deepcopy(initial), copy.deepcopy(initial))
    assert stopped["status"] == "deadline"
    assert stopped["counters"]["target_prediction_tokens"] == 0
    assert Path(stopped["checkpoint_path"]).exists()
    resumed = train(dataclasses.replace(deadline_config, deadline_utc=None,
                    resume_from=stopped["checkpoint_path"]), copy.deepcopy(initial), copy.deepcopy(initial))
    assert resumed["status"] == "completed"
    assert resumed["counters"]["target_prediction_tokens"] == 113


def test_bank_capacity_and_unique_coverage_are_separate(tmp_path):
    path = tmp_path / "duplicate_bank.pt"
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
    torch.save({"input_ids": ids}, path)
    data = load_training_data(path)
    assert data["prediction_tokens"] == 9
    assert data["unique_prediction_tokens"] == 6
    assert data["unique_chunks"] == 2


def test_duplicate_chunk_coverage_unions_prediction_prefixes_exactly(tmp_path):
    path=tmp_path/"duplicates.pt"
    torch.save({"input_ids":torch.tensor([[1,2,3,4],[1,2,3,4],[4,3,2,1]])},path)
    stream=TokenStream(load_training_data(path),seed=11)
    stream.order=[0,1,2]
    list(stream.take(4,2))
    assert stream.unique_prediction_tokens_exposed()==3
    list(stream.take(3,2))
    assert stream.unique_prediction_tokens_exposed()==4
    list(stream.take(2,2))
    assert stream.epoch==1 and stream.unique_prediction_tokens_exposed()==6
    list(stream.take(4,2))
    coverage=stream.coverage_accounting()
    assert coverage["unique_prediction_tokens_exposed"]==6
    assert coverage["repeated_prediction_token_exposures"]==7
    assert not coverage["assumption_no_duplicate_chunks"]
