#!/usr/bin/env python3
"""Run a JSON-configured independent full-model replay experiment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.train import TrainConfig, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Path to a full training checkpoint")
    parser.add_argument("--device", help="Override device, e.g. cuda:0 or cpu")
    arguments = parser.parse_args()
    config = TrainConfig.from_dict(json.loads(Path(arguments.config).read_text()))
    if arguments.resume:
        config.resume_from = arguments.resume
    if arguments.device:
        config.device = arguments.device
    result = train(config)
    print(json.dumps({"status": result["status"], "output_dir": config.output_dir,
                      "counters": result["counters"]}, sort_keys=True))


if __name__ == "__main__":
    main()
