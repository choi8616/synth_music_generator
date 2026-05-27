from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from task1_musicgen.data_pipeline import load_sequences, load_tokenizer
from task1_musicgen.lstm_model import LSTMConfig, LSTMLanguageModel, TokenWindowDataset, choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LSTM next-token model.")
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--model-out", default="outputs/lstm_model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float | None:
    if len(loader.dataset) == 0:
        return None

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    tokenizer = load_tokenizer(processed_dir)
    train_sequences = load_sequences(processed_dir / "train_sequences.npy")
    val_sequences = load_sequences(processed_dir / "val_sequences.npy")

    train_dataset = TokenWindowDataset(
        train_sequences,
        context_length=args.context_length,
        stride=args.stride,
        max_windows=args.max_train_windows,
    )
    val_dataset = TokenWindowDataset(
        val_sequences,
        context_length=args.context_length,
        stride=args.stride,
        max_windows=args.max_val_windows,
    )

    if len(train_dataset) == 0:
        raise SystemExit("No training windows. Try a smaller --context-length.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = choose_device(args.device)
    config = LSTMConfig(
        vocab_size=len(tokenizer),
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = LSTMLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    print(f"Device: {device}")
    print(f"Train windows: {len(train_dataset)}")
    print(f"Val windows:   {len(val_dataset)}")
    print(f"Vocab size:    {len(tokenizer)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss}")

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "config": config.__dict__,
        "history": history,
        "special_tokens": {
            "BOS_None": tokenizer.vocab.get("BOS_None"),
            "EOS_None": tokenizer.vocab.get("EOS_None"),
            "PAD_None": tokenizer.vocab.get("PAD_None"),
            "MASK_None": tokenizer.vocab.get("MASK_None"),
        },
    }
    torch.save(checkpoint, out_path)

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"history": history, "config": checkpoint["config"]}, indent=2), encoding="utf-8")
    print(f"Saved LSTM checkpoint: {out_path.resolve()}")


if __name__ == "__main__":
    main()

