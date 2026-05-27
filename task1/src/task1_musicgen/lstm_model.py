from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LSTMConfig:
    vocab_size: int
    context_length: int = 128
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2


class TokenWindowDataset(Dataset):
    """Create fixed-length next-token training windows from token sequences.

    For a token sequence [a, b, c, d], the model input could be [a, b, c]
    and the target would be [b, c, d]. That trains the model to predict the
    next token at every position.
    """

    def __init__(
        self,
        sequences: list[list[int]],
        context_length: int = 128,
        stride: int | None = None,
        max_windows: int | None = None,
    ) -> None:
        self.sequences = sequences
        self.context_length = context_length
        self.stride = stride or context_length
        self.windows: list[tuple[int, int]] = []

        for seq_idx, seq in enumerate(sequences):
            if len(seq) <= context_length:
                continue
            for start in range(0, len(seq) - context_length, self.stride):
                self.windows.append((seq_idx, start))
                if max_windows is not None and len(self.windows) >= max_windows:
                    return

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_idx, start = self.windows[index]
        seq = self.sequences[seq_idx]
        end = start + self.context_length
        x = torch.tensor(seq[start:end], dtype=torch.long)
        y = torch.tensor(seq[start + 1 : end + 1], dtype=torch.long)
        return x, y


class LSTMLanguageModel(nn.Module):
    """A small next-token LSTM language model for symbolic music tokens."""

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        hidden_states, _ = self.lstm(embeddings)
        logits = self.output(hidden_states)
        return logits


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = 20,
    forbidden_ids: set[int] | None = None,
) -> int:
    """Sample one token id from the model's final-position logits."""

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits.float().clone()
    if forbidden_ids:
        for token_id in forbidden_ids:
            if 0 <= token_id < logits.numel():
                logits[token_id] = -float("inf")

    logits = logits / temperature
    if top_k is not None and top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, -float("inf"))
        filtered[indices] = values
        logits = filtered

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_tokens(
    model: LSTMLanguageModel,
    start_ids: list[int],
    max_new_tokens: int = 512,
    eos_id: int | None = None,
    forbidden_ids: set[int] | None = None,
    temperature: float = 1.0,
    top_k: int | None = 20,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> list[int]:
    """Autoregressively generate token ids from a trained LSTM."""

    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device(device)
    model.eval()
    model.to(device)

    generated = list(start_ids)
    context_length = model.config.context_length

    for _ in range(max_new_tokens):
        context = generated[-context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        next_id = sample_from_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            forbidden_ids=forbidden_ids,
        )
        if next_id == eos_id:
            break
        generated.append(next_id)

    return generated

