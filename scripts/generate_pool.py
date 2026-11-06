#!/usr/bin/env python3
"""Generate one seed's fixed-length shared teacher pool from a JSON protocol."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.generate import GenerationConfig, generate_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.config).read_text())
    if args.resume:
        payload["resume"] = True
    config = GenerationConfig(**payload)
    result = generate_pool(config, progress=lambda event: print(json.dumps(event), flush=True))
    print(json.dumps({"output": config.output, "wall_seconds": result["generation_wall_seconds"],
                      "prediction_tokens": result["prediction_tokens"], "diagnostics": result["diagnostics"]}), flush=True)


if __name__ == "__main__":
    main()
