from __future__ import annotations

import argparse
from pathlib import Path

import torch

from task1_musicgen.data_pipeline import decode_ids_to_midi, load_tokenizer
from task1_musicgen.lstm_model import LSTMConfig, LSTMLanguageModel, choose_device, generate_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MIDI from a trained LSTM model.")
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--model-path", default="outputs/lstm_model.pt")
    parser.add_argument("--out-midi", default="outputs/symbolic_unconditioned_lstm.mid")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.processed_dir)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    config = LSTMConfig(**checkpoint["config"])
    model = LSTMLanguageModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    special = checkpoint.get("special_tokens", {})
    bos_id = special.get("BOS_None", tokenizer.vocab.get("BOS_None"))
    eos_id = special.get("EOS_None", tokenizer.vocab.get("EOS_None"))
    forbidden_ids = {
        token_id
        for token_id in [
            special.get("PAD_None", tokenizer.vocab.get("PAD_None")),
            special.get("MASK_None", tokenizer.vocab.get("MASK_None")),
        ]
        if token_id is not None
    }

    start_ids = [bos_id] if bos_id is not None else []
    device = choose_device(args.device)
    generated = generate_tokens(
        model,
        start_ids=start_ids,
        max_new_tokens=args.max_new_tokens,
        eos_id=eos_id,
        forbidden_ids=forbidden_ids,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        device=device,
    )
    ids_for_midi = [token_id for token_id in generated if token_id != bos_id]
    out_path = decode_ids_to_midi(tokenizer, ids_for_midi, args.out_midi)

    print("Generated LSTM sample.")
    print(f"  Tokens: {len(ids_for_midi)}")
    print(f"  MIDI:   {Path(out_path).resolve()}")


if __name__ == "__main__":
    main()

