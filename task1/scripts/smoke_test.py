from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quickly inspect processed Task 1 artifacts.")
    parser.add_argument("--processed-dir", default="processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    required = [
        "tokenizer.json",
        "train_sequences.npy",
        "val_sequences.npy",
        "train_tokens.npy",
        "val_tokens.npy",
        "vocab_info.json",
        "tokenization_report.json",
    ]

    missing = [name for name in required if not (processed_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing processed files: {missing}")

    report = json.loads((processed_dir / "tokenization_report.json").read_text(encoding="utf-8"))
    print("Processed data looks present.")
    print(f"  MIDI files:      {report['n_midi_files']}")
    print(f"  Train sequences: {report['n_train_sequences']}")
    print(f"  Val sequences:   {report['n_val_sequences']}")
    print(f"  Failed files:    {report['n_failed_files']}")
    print(f"  Vocab size:      {report['vocab_info']['vocab_size']}")
    print(f"  Median length:   {report['sequence_length']['median']}")


if __name__ == "__main__":
    main()

