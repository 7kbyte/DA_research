#!/usr/bin/env python3
"""Summarize sampled GPU telemetry without equating utilization with FLOPs."""
import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path


def summarize(records, maximum_gap=90):
    devices = collections.defaultdict(list)
    for row in records:
        if row.get("returncode") != 0:
            continue
        for gpu in row.get("gpus", []):
            try:
                values = {key: float(gpu[key]) for key in
                          ("utilization_percent", "memory_mib", "power_w", "temperature_c")}
                devices[gpu["index"]].append({"epoch": row["epoch"], "uuid": gpu["uuid"], **values})
            except (KeyError, ValueError):
                continue
    results = []
    for index, samples in sorted(devices.items()):
        samples.sort(key=lambda x: x["epoch"])
        covered, utilization_seconds, energy_joules, skipped = 0., 0., 0., 0.
        for a, b in zip(samples, samples[1:]):
            duration = b["epoch"] - a["epoch"]
            if duration <= 0 or duration > maximum_gap or a["uuid"] != b["uuid"]:
                skipped += max(0, duration)
                continue
            covered += duration
            utilization_seconds += duration * (a["utilization_percent"] + b["utilization_percent"]) / 200
            energy_joules += duration * (a["power_w"] + b["power_w"]) / 2
        results.append({"gpu": index, "uuid": samples[0]["uuid"], "samples": len(samples),
                        "observed_span_seconds": samples[-1]["epoch"] - samples[0]["epoch"],
                        "integrated_coverage_seconds": covered, "skipped_gap_seconds": skipped,
                        "time_weighted_mean_utilization_percent": utilization_seconds / covered * 100 if covered else None,
                        "utilization_integral_gpu_hours": utilization_seconds / 3600,
                        "sampled_gpu_energy_kwh": energy_joules / 3_600_000,
                        "maximum_sampled_memory_mib": max(s["memory_mib"] for s in samples),
                        "maximum_sampled_temperature_c": max(s["temperature_c"] for s in samples)})
    return results, devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("logs/gpu_telemetry.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/resources"))
    parser.add_argument("--maximum-gap-seconds", type=float, default=90)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    rows, incomplete = [], 0
    for line in raw.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            incomplete += 1
    devices, samples = summarize(rows, args.maximum_gap_seconds)
    result = {"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "source": str(args.input), "source_sha256": hashlib.sha256(raw).hexdigest(),
              "records": len(rows), "incomplete_lines_skipped": incomplete,
              "integration": "Trapezoidal over adjacent successful samples, excluding gaps above threshold",
              "maximum_gap_seconds": args.maximum_gap_seconds,
              "scope": "Allocated GPU telemetry, not per-process attribution or total system energy",
              "interpretation": "Utilization integral is a sampled activity proxy, not training GPU-hours or FLOPs. Unobserved times are not counted as idle.",
              "devices": devices,
              "total_sampled_gpu_energy_kwh": sum(x["sampled_gpu_energy_kwh"] for x in devices),
              "total_utilization_integral_gpu_hours": sum(x["utilization_integral_gpu_hours"] for x in devices)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "telemetry_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if samples and not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        start = min(s[0]["epoch"] for s in samples.values())
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for index, data in sorted(samples.items()):
            hours = [(s["epoch"] - start) / 3600 for s in data]
            axes[0].plot(hours, [s["utilization_percent"] for s in data], label=f"GPU {index}", lw=.8)
            axes[1].plot(hours, [s["power_w"] for s in data], label=f"GPU {index}", lw=.8)
        axes[0].set(ylabel="Sampled GPU utilization (%)", ylim=(-2, 102))
        axes[0].legend(ncol=4)
        axes[1].set(ylabel="GPU power (W)", xlabel="Hours since first telemetry sample")
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle("Observed GPU activity (sampled; not FLOPs or process attribution)")
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(args.output / f"gpu_activity.{suffix}", dpi=160)
        plt.close(fig)
    print(json.dumps({"output": str(args.output), "records": len(rows), "gpus": len(devices)}))


if __name__ == "__main__":
    main()
