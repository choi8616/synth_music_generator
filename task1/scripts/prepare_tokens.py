from __future__ import annotations

import argparse
from pathlib import Path

from task1_musicgen.data_pipeline import (
    TokenizationConfig,
    build_tokenizer,
    collect_midi_files,
    flatten_sequences,
    save_sequences,
    save_tokenization_report,
    save_vocab_info,
    tokenize_midi_files,
    train_val_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn a folder of MIDI files into train/validation REMI token sequences."
    )
    parser.add_argument("--data-dir", required=True, help="Folder containing .mid/.midi files.")
    parser.add_argument("--processed-dir", default="processed", help="Where processed files are written.")
    parser.add_argument("--vocab-size", type=int, default=1000) # max vocab size for tokens
    parser.add_argument("--val-ratio", type=float, default=0.1) # train vs validation ratio
    parser.add_argument("--seed", type=int, default=42) # random seed for train/val split
    parser.add_argument("--max-files", type=int, default=None) # max number of midi files to use
    parser.add_argument("--min-tokens", type=int, default=16) # min number of tokens
    parser.add_argument(
        "--use-programs",
        action="store_true",
        help="Include MIDI program/instrument tokens. Leave off for the first simple pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    config = TokenizationConfig(
        data_dir=Path(args.data_dir),
        processed_dir=processed_dir,
        vocab_size=args.vocab_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_files=args.max_files,
        min_tokens=args.min_tokens,
        use_programs=args.use_programs,
    )

    midi_files = collect_midi_files(config.data_dir, config.max_files)
    print(f"Found {len(midi_files)} MIDI files")
    if not midi_files:
        raise SystemExit("No MIDI files found. Check --data-dir.")

    print("Training REMI tokenizer...")
    tokenizer = build_tokenizer(midi_files, config)
    tokenizer.save(processed_dir, filename="tokenizer.json")

    print("Tokenizing MIDI files...")
    sequences, failures = tokenize_midi_files(midi_files, tokenizer, min_tokens=config.min_tokens)
    if not sequences:
        raise SystemExit("No usable token sequences were produced.")

    train_sequences, val_sequences = train_val_split(sequences, config.val_ratio, config.seed)

    save_sequences(processed_dir / "train_sequences.npy", train_sequences)
    save_sequences(processed_dir / "val_sequences.npy", val_sequences)

    eos_id = tokenizer.vocab.get("EOS_None")
    flatten_train = flatten_sequences(train_sequences, separator_id=eos_id)
    flatten_val = flatten_sequences(val_sequences, separator_id=eos_id)
    import numpy as np

    np.save(processed_dir / "train_tokens.npy", flatten_train)
    np.save(processed_dir / "val_tokens.npy", flatten_val)

    vocab_info = save_vocab_info(tokenizer, processed_dir / "vocab_info.json")
    save_tokenization_report(
        processed_dir / "tokenization_report.json",
        config,
        midi_files,
        train_sequences,
        val_sequences,
        failures,
        vocab_info,
    )

    print("Done.")
    print(f"  Train sequences: {len(train_sequences)}")
    print(f"  Val sequences:   {len(val_sequences)}")
    print(f"  Failures:        {len(failures)}")
    print(f"  Vocab size:      {vocab_info['vocab_size']}")
    print(f"  Wrote files to:  {processed_dir.resolve()}")


if __name__ == "__main__":
    main()

