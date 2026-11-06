# Engineering contract (initial; pilot may revise before formal freeze)

Workspace: `/data/202610/7kb/research`. Deadline: **2026-11-07 03:26:32 UTC**.
Scratch: `/tmp/grad-research-20261106` (large model/data/checkpoint caches).
Use Python 3.10, existing CUDA PyTorch 2.3, Hugging Face GPTNeoX/Pythia.
Do not replace CUDA PyTorch. Pin dependencies and model/dataset revisions.

## Scope and ownership

- `src/replay_study/data.py`, `scripts/prepare_data.py`, `scripts/download_assets.py`: data agent.
- `src/replay_study/train.py`, `src/replay_study/objectives.py`, `scripts/run_train.py`: training agent.
- `src/replay_study/evaluate.py`, `src/replay_study/selection.py`, `src/replay_study/generate.py`, matching CLI scripts: evaluation/selection agent.
- Root agent owns documentation, experiment registry/orchestration, analysis, shared config and integration.
- Each agent owns its related tests; prefix filenames accordingly. Do not edit another agent's files without coordination.

## Shared data format

`artifacts/data/<name>.pt` is a `torch.save` dict with:
`input_ids`: contiguous CPU `torch.int32` or `torch.long` tensor `[N, sequence_length]`.
Optional `attention_mask`: bool or integer tensor, same shape; missing means all valid.
Optional `doc_ids`: list of N document IDs (for chunk provenance).
Always an adjacent `.json` manifest: source, revision, split, tokenizer, token/document counts, hashes, preprocessing.
Keep document splits disjoint before chunking. Never use validation/test examples for replay/selection/generation prompts.
Prefer independent per-document chunks; document-boundary policy must be explicit. Labels use causal shift, valid prediction-token counts are L-1 for all-valid chunks.
Target: FineMath finemath-4plus. General replay/eval: C4 en. Independent general eval: WikiText-103.
Initial sequence length: **512**. Initial target unique tokens about 2M; target validation/test at least 128K tokens each. General eval validation/test at least 128K each where feasible. General candidate pool at least 1M tokens. Candidate/replay sources must be separate from evaluation. Save doc hashes and verify split duplicates.
Models live in `/tmp/grad-research-20261106/models`, or HF cache under scratch. Primary `EleutherAI/pythia-410m` final checkpoint; debugging optional pythia-160m.

## Objectives and experiments

B0: target CE only. B1 C4 CE, B2 C4 forward KL, B3 generated random CE, B4 generated random forward KL, D generated dedup random KL, M generated dedup five-quantile teacher-NLL stratified KL, C generated cluster representative KL.
Full-model updates, frozen initial teacher, valid-token normalized target/replay losses, teacher-to-student KL. Exact full-vocabulary KL (chunk vocabulary/time dimension to control memory if necessary). Preserve identical target stream and optimizer schedule within seed.
Unique bank size K, replay exposures T_R, target exposures T_new, b=T_R/T_new, lambda, and all preprocessing/evaluation costs are separate fields. Token budgets mean prediction tokens (causal shift) and additionally report raw input token counts.
Initial formal candidates: seeds 11, 22, 33; 10M target prediction-token exposures; ~100K unique replay prediction tokens; b=.05; sensitivity b=.01/.10 for B4/M. Pilot controls schedule and hyperparameters before formal freeze.
Each run has JSON config, JSONL training events, intermediate/final validation metrics, provenance, wall time and peak GPU memory. Checkpoints selected on validation only; test evaluation deferred to frozen protocol.
Evaluation reports per-document/per-chunk NLL sums/counts and teacher KL sums/counts where feasible, separately by corpus, without implying reasoning capability.

## Collaboration

Send root API/CLI examples and progress. Report network or incompatibility blockers immediately. Keep dependencies lightweight; coordinate install with data agent. Avoid unbounded jobs and reserve enough time for final evaluation/packaging.
