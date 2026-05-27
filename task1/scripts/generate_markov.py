from __future__ import annotations

import argparse
from pathlib import Path

from task1_musicgen.data_pipeline import decode_ids_to_midi, load_tokenizer
from task1_musicgen.markov import NGramModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MIDI from a trained Markov baseline.")
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--model-path", default="outputs/markov_order4.pkl")
    parser.add_argument("--out-midi", default="outputs/symbolic_unconditioned_markov.mid")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.processed_dir)
    model = NGramModel.load(args.model_path)
    ids = model.generate(
        max_length=args.max_length,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    out_path = decode_ids_to_midi(tokenizer, ids, args.out_midi)

    print("Generated Markov sample.")
    print(f"  Tokens: {len(ids)}")
    print(f"  MIDI:   {Path(out_path).resolve()}")


if __name__ == "__main__":
    main()

