#!/usr/bin/env python3
"""CPU-only exact FP32 XOR/byte-plane model weight delivery helpers."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.weight_archive import encode_weights, decode_weights, verify_roundtrip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("encode")
    encode.add_argument("--checkpoint", required=True)
    encode.add_argument("--base", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--chunk-mib", type=int, default=4)
    encode.add_argument("--level", type=int, default=6)
    encode.add_argument("--expected-model-sha256")
    decode = commands.add_parser("decode")
    decode.add_argument("--archive", required=True)
    decode.add_argument("--base", required=True)
    decode.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--checkpoint", required=True)
    verify.add_argument("--restored", required=True)
    verify.add_argument("--archive", required=True)
    verify.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "encode":
        report = encode_weights(args.checkpoint, args.base, args.output, threads=args.threads,
            chunk_bytes=args.chunk_mib << 20, compression_level=args.level,
            expected_model_sha256=args.expected_model_sha256)
    elif args.command == "decode":
        report = decode_weights(args.archive, args.base, args.output, threads=args.threads)
    else:
        report = verify_roundtrip(args.checkpoint, args.restored, args.archive, threads=args.threads)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
