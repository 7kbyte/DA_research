import json
from pathlib import Path
import torch

for name in ("main_B4_seed11_final", "A_B4_lambda5_seed11_final"):
    saved = torch.load(f"restored/{name}.pt", map_location="cpu",
                       mmap=True, weights_only=False)
    provenance = saved["lossless_delivery_provenance"]
    print(name, json.dumps({
        "model_fingerprint": provenance["model_fingerprint"],
        "checkpoint_kind": saved["checkpoint_kind"],
        "source_checkpoint_kind": provenance["source_checkpoint_kind"],
        "source_checkpoint_sha256": provenance["source_checkpoint_sha256"],
    }, indent=2))

main = json.loads(Path("primary_final_weights_manifest.json").read_text())
main_row = next(r for r in main["logical_runs"]
                if r["condition_run_id"] == "formal_deterministic_B4_s11_b0p05_w0p05")
print("Main own metadata:", main_row["original_checkpoint_metadata"])
sidecar = torch.load(
    "metadata/lambda_sweep_deterministic_B4_s11_b0p05_w5__validation_selected_lambda_final.source_metadata.pt",
    map_location="cpu", weights_only=False)
print("A own logical identity:", sidecar["logical_identity"])
print("A original explicit checkpoint_kind:",
      sidecar["original_checkpoint_metadata"].get("checkpoint_kind", "<not explicitly stored>"))
