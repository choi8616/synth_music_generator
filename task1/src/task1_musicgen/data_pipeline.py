from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from miditok import REMI, TokSequence, TokenizerConfig
from symusic import Score


MIDI_EXTENSIONS = {".mid", ".midi"}


@dataclass(frozen=True)
class TokenizationConfig:
    """Settings for turning a folder of MIDI files into token sequences."""

    data_dir: Path
    processed_dir: Path
    vocab_size: int = 1000
    val_ratio: float = 0.1
    seed: int = 42
    max_files: int | None = None
    min_tokens: int = 16
    use_programs: bool = False


def collect_midi_files(data_dir: str | Path, max_files: int | None = None) -> list[Path]:
    """Return all MIDI files under data_dir, sorted for reproducibility."""

    data_dir = Path(data_dir).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"MIDI folder does not exist: {data_dir}")

    files = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MIDI_EXTENSIONS
    )
    if max_files is not None:
        files = files[:max_files]
    return files


def build_tokenizer(midi_files: list[Path], config: TokenizationConfig) -> REMI:
    """Train a REMI tokenizer on the MIDI files.

    REMI represents symbolic music as language-like tokens such as Bar, Position,
    Pitch, Velocity, and Duration. This is the common token stream used by both
    the Markov baseline and the LSTM language model.
    """

    if not midi_files:
        raise ValueError("No MIDI files were provided for tokenizer training.")

    tokenizer_config = TokenizerConfig(
        num_velocities=1,
        use_chords=False,
        use_programs=config.use_programs,
    )
    tokenizer = REMI(tokenizer_config)
    tokenizer.train(vocab_size=config.vocab_size, files_paths=midi_files)
    return tokenizer


def _as_tok_sequences(encoded) -> list[TokSequence]:
    """Normalize MidiTok output to a list of TokSequence objects."""

    if isinstance(encoded, TokSequence):
        return [encoded]
    if isinstance(encoded, list):
        return [seq for seq in encoded if isinstance(seq, TokSequence)]
    raise TypeError(f"Unexpected tokenizer output type: {type(encoded)!r}")


def tokenize_midi_file(
    midi_file: str | Path,
    tokenizer: REMI,
    min_tokens: int = 16,
) -> list[list[int]]:
    """Tokenize one MIDI file and return one or more integer token sequences."""

    score = Score(Path(midi_file))
    encoded = tokenizer(score)
    sequences: list[list[int]] = []

    for tokseq in _as_tok_sequences(encoded):
        ids = list(tokseq.ids)
        if len(ids) >= min_tokens:
            sequences.append(ids)

    return sequences


def tokenize_midi_files(
    midi_files: Iterable[Path],
    tokenizer: REMI,
    min_tokens: int = 16,
) -> tuple[list[list[int]], list[dict]]:
    """Tokenize MIDI files, returning successful sequences and failure records."""

    sequences: list[list[int]] = []
    failures: list[dict] = []

    for midi_file in midi_files:
        try:
            sequences.extend(tokenize_midi_file(midi_file, tokenizer, min_tokens))
        except Exception as exc:  # MIDI corpora often contain a few broken files.
            failures.append({"file": str(midi_file), "error": repr(exc)})

    return sequences, failures


def train_val_split(
    sequences: list[list[int]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[list[int]], list[list[int]]]:
    """Shuffle and split token sequences."""

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    rng = random.Random(seed)
    shuffled = list(sequences)
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_sequences = shuffled[:n_val]
    train_sequences = shuffled[n_val:]
    return train_sequences, val_sequences


def save_sequences(path: str | Path, sequences: list[list[int]]) -> None:
    """Save variable-length token sequences as a numpy object array."""

    array = np.array([np.array(seq, dtype=np.int64) for seq in sequences], dtype=object)
    np.save(Path(path), array, allow_pickle=True)


def load_sequences(path: str | Path) -> list[list[int]]:
    """Load variable-length token sequences saved by save_sequences."""

    array = np.load(Path(path), allow_pickle=True)
    return [seq.astype(np.int64).tolist() for seq in array]


def flatten_sequences(
    sequences: list[list[int]],
    separator_id: int | None = None,
) -> np.ndarray:
    """Flatten sequences into one stream, optionally inserting a separator token."""

    flat: list[int] = []
    for seq in sequences:
        flat.extend(seq)
        if separator_id is not None:
            flat.append(separator_id)
    return np.array(flat, dtype=np.int64)


def save_vocab_info(tokenizer: REMI, path: str | Path) -> dict:
    """Save lightweight vocabulary metadata for analysis and debugging."""

    vocab = tokenizer.vocab
    info = {
        "vocab_size": len(tokenizer),
        "base_vocab_size": len(vocab),
        "special_tokens": {
            token: idx
            for token, idx in vocab.items()
            if token.endswith("_None") and token.split("_", 1)[0] in {"PAD", "BOS", "EOS", "MASK"}
        },
        "token_type_counts": {},
        "first_tokens": list(vocab.items())[:25],
    }

    for token in vocab:
        token_type = token.split("_", 1)[0]
        info["token_type_counts"][token_type] = info["token_type_counts"].get(token_type, 0) + 1

    Path(path).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def save_tokenization_report(
    path: str | Path,
    config: TokenizationConfig,
    midi_files: list[Path],
    train_sequences: list[list[int]],
    val_sequences: list[list[int]],
    failures: list[dict],
    vocab_info: dict,
) -> None:
    """Write a JSON report so the notebook/presentation can explain the data step."""

    lengths = [len(seq) for seq in train_sequences + val_sequences]
    report = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "n_midi_files": len(midi_files),
        "n_sequences_total": len(train_sequences) + len(val_sequences),
        "n_train_sequences": len(train_sequences),
        "n_val_sequences": len(val_sequences),
        "n_failed_files": len(failures),
        "failures": failures[:25],
        "sequence_length": {
            "min": int(np.min(lengths)) if lengths else 0,
            "median": float(np.median(lengths)) if lengths else 0.0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "max": int(np.max(lengths)) if lengths else 0,
        },
        "vocab_info": vocab_info,
    }
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_tokenizer(processed_dir: str | Path) -> REMI:
    """Load the tokenizer saved by prepare_tokens.py."""

    tokenizer_path = Path(processed_dir) / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer_path}")
    return REMI(params=tokenizer_path)


def decode_ids_to_midi(
    tokenizer: REMI,
    ids: list[int],
    out_path: str | Path,
    encoded_ids: bool = True,
) -> Path:
    """Decode generated token ids into a MIDI file."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokseq = TokSequence(ids=list(ids), are_ids_encoded=encoded_ids)
    score = tokenizer([tokseq])
    score.dump_midi(out_path)
    return out_path
