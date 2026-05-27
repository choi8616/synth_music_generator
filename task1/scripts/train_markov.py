from __future__ import annotations

import argparse
import json
from pathlib import Path

from task1_musicgen.data_pipeline import load_sequences, load_tokenizer
from task1_musicgen.markov import NGramModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the n-gram Markov baseline.")
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--model-out", default="outputs/markov_order4.pkl")
    parser.add_argument("--order", type=int, default=4, help="4 means p(next | previous 3 tokens).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    tokenizer = load_tokenizer(processed_dir)
    train_sequences = load_sequences(processed_dir / "train_sequences.npy")
    val_sequences = load_sequences(processed_dir / "val_sequences.npy")

    bos_id = tokenizer.vocab.get("BOS_None")
    eos_id = tokenizer.vocab.get("EOS_None")

    model = NGramModel(order=args.order, bos_id=bos_id, eos_id=eos_id)
    model.fit(train_sequences)
    val_perplexity = model.perplexity(val_sequences) if val_sequences else None
    model.save(args.model_out)

    metrics = {
        "order": args.order,
        "n_contexts": len(model.counts),
        "n_train_sequences": len(train_sequences),
        "n_val_sequences": len(val_sequences),
        "val_perplexity": val_perplexity,
        "model_path": str(Path(args.model_out).resolve()),
    }
    metrics_path = Path(args.model_out).with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Trained Markov baseline.")
    print(f"  Order:          {args.order}")
    print(f"  Contexts:       {len(model.counts)}")
    print(f"  Val perplexity: {val_perplexity}")
    print(f"  Saved model:    {Path(args.model_out).resolve()}")


if __name__ == "__main__":
    main()

