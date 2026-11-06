#!/usr/bin/env python3
"""Sample allocated-GPU telemetry until the research deadline (no mutations)."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deadline", required=True)
    p.add_argument("--output", type=Path, default=Path("logs/gpu_telemetry.jsonl"))
    p.add_argument("--interval", type=float, default=30)
    args = p.parse_args()
    deadline = dt.datetime.fromisoformat(args.deadline.replace("Z", "+00:00")).timestamp()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    query = ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu", "--format=csv,noheader,nounits"]
    while time.time() < deadline:
        result = subprocess.run(query, capture_output=True, text=True, timeout=20)
        fields = ["index", "uuid", "utilization_percent", "memory_mib", "power_w", "temperature_c"]
        gpus = [dict(zip(fields, [v.strip() for v in line.split(",")])) for line in result.stdout.splitlines()]
        record = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "epoch": time.time(), "gpus": gpus,
                  "returncode": result.returncode, "error": result.stderr.strip(), "loadavg": os.getloadavg()}
        with args.output.open("a") as f:
            f.write(json.dumps(record) + "\n")
        time.sleep(min(args.interval, 30))


if __name__ == "__main__":
    main()
