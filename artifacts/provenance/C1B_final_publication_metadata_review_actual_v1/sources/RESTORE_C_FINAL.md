# Extended C final weights

Only the complete frozen model-specific final cohort is in this package. Every independently encoded model passed bytewise comparison with its source checkpoint. D aliases exist only with explicit source-bound exact final-state equality. Each alias retains its own full metadata sidecar.

For 410M, verify the primary_final_weights_lossless_v1.tar SHA in base_dependency, then extract its base/pythia-410m-final directory. For 1B/2.8B, native base/config/tokenizer/README are included under base/. No network download is needed.

```bash
python tools/scripts/archive_model_weights.py --threads 2 decode --archive weights/RUN.xor.zip --base base/MODEL-final/model.safetensors --output restored.pt
```

For D aliases decode the referenced B4 weight member and use D own metadata sidecar for logical identity/provenance/exposure. The restored file contains model weights; full original nonweight/nonoptimizer metadata is preserved separately in metadata/*.source_metadata.pt. Optimizer state is excluded; full Adam resume is not claimed. Source checkpoints, ZIPs and verification outputs are retained in scratch. A separate durable publication receipt is required.
