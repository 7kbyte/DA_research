#!/usr/bin/env python3
"""Reuse frozen 410M generated banks and sample C4 from the expanded train pool."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.selection import save_bank


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path("artifacts/extended/pythia-410m/banks")
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "pythia-410m", "generated_banks": "exact reuse of primary frozen banks",
        "reason": "Initial teacher, generation protocol, seeds and bank size are unchanged.",
        "c4_pool": "artifacts/data_extended/c4_candidate.pt",
        "evaluation_requirement": "Use data_extended heldout sets with expanded training.",
        "new_generation_cost_seconds": 0, "bank_chunks": 196, "banks": [],
    }
    for seed in (101, 11, 22, 33):
        for method in ("random", "dedup_random", "nll_stratified", "cluster"):
            source = Path(f"artifacts/banks/generated_{method}_seed{seed}.pt")
            destination = root / source.name
            if destination.exists() and sha256(destination) != sha256(source):
                raise ValueError(f"Existing artifact conflicts with frozen source: {destination}")
            shutil.copyfile(source, destination)
            metadata = json.loads(source.with_suffix(".json").read_text())
            assert sha256(destination) == metadata["artifact_sha256"]
            metadata["extended_reuse_provenance"] = {
                "source_artifact": str(source),
                "source_manifest_sha256": sha256(source.with_suffix(".json")),
                "byte_identical_to_primary": True,
                "additional_generation_or_selection_seconds": 0,
            }
            destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            summary["banks"].append({"seed": seed, "method": method,
                                     "artifact": str(destination), "sha256": sha256(destination)})
        destination = root / f"c4_random_seed{seed}.pt"
        metadata = save_bank(summary["c4_pool"], destination, method="random", seed=seed, bank_size=196)
        summary["banks"].append({"seed": seed, "method": "c4_random", "artifact": str(destination),
                                 "sha256": metadata["artifact_sha256"],
                                 "selection_seconds": metadata["wall_seconds"]})
    Path("results/extended_pythia-410m_bank_provenance.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"Prepared {len(summary['banks'])} banks; generated artifacts are byte-identical to primary.")


if __name__ == "__main__":
    main()
