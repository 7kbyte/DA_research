#!/usr/bin/env python3
"""Read only scheduling, execution timing and hashes; never parse scored outputs."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from evaluation_plan import encoded, sha


STATE_FIELDS = ("status", "gpu", "pid", "started_at_utc", "ended_at_utc", "elapsed_seconds", "returncode", "missing_outputs")


def inspect(state_path, manifest_path, authority_path):
    authority = json.loads(Path(authority_path).read_text())
    if sha(manifest_path) != authority["manifest_sha256"]:
        raise ValueError("Timing authority and manifest SHA differ")
    manifest = json.loads(Path(manifest_path).read_text())
    job = next(j for j in manifest["jobs"] if j["id"] == authority["job_id"])
    if job["kind"] != "initial":
        raise ValueError("This timing authorization permits only the initial model")
    state_bytes = Path(state_path).read_bytes()
    state = json.loads(state_bytes)
    record = state["jobs"][job["id"]]
    result = {"recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "job_id": job["id"],
        "state_snapshot": {k: record[k] for k in STATE_FIELDS if k in record},
        "manifest": str(manifest_path), "manifest_sha256": authority["manifest_sha256"],
        "authority": str(authority_path), "authority_sha256": sha(authority_path),
        "source_state_sha256_at_read": hashlib.sha256(state_bytes).hexdigest(), "scores_or_predictions_parsed": False,
        "model_inference_started_by_inspector": False, "complete_timing_available": False}
    path = Path(job["output"]); attest_path = path.with_name(path.stem + "_execution.json")
    if attest_path.exists():
        attest = json.loads(attest_path.read_text())
        if (attest.get("manifest_sha256") != authority["manifest_sha256"] or attest.get("job_id") != job["id"]
                or attest.get("verified_frozen_inputs") is not True or attest.get("checkpoint_sha256") is not None):
            raise ValueError("Initial timing attestation does not match frozen authorization")
        # Read bytes solely for hashing; do not deserialize summary/prediction files.
        integrity = {source: sha(source) == expected for source, expected in attest["output_sha256"].items()}
        if not all(integrity.values()) or set(integrity) != {str(path), *[
                str(path.with_name(path.stem + f"_{name}.jsonl")) for name in manifest["baseline_groups"][job["baseline_group"]]["tasks"]]}:
            raise ValueError("Expected summary and both full-split prediction files must match attested hashes")
        result["execution"] = {k: attest[k] for k in ("verification_wall_seconds", "inference_subprocess_wall_seconds",
            "started_at_utc", "finished_at_utc", "preflight_environment") if k in attest}
        result.update(attestation=str(attest_path), attestation_sha256=sha(attest_path), output_hashes_valid=True,
                      expected_output_files_verified=len(integrity),
                      complete_timing_available=record.get("status") == "completed" and record.get("returncode") == 0)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, default=Path("logs/supplemental_job_state.json"))
    p.add_argument("--manifest", type=Path, default=Path("docs/primary_evaluation_v2_execution_matched_behavior_manifest.json"))
    p.add_argument("--authority", type=Path, default=Path("docs/first_behavior_timing_execution.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = inspect(args.state, args.manifest, args.authority)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
