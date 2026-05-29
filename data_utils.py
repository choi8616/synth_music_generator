"""
Synth-pop Music Generation — Data Utilities
============================================

Shared functions for loading the curated synth-pop MIDI dataset.
Lives at the repo root; reads from ./processed/.

Usage from a notebook in a subfolder (e.g. data_extraction/ or task2/):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))   # add repo root to path
    from data_utils import dataset_summary, iter_melodies, iter_drums, iter_bass
"""

import pretty_midi
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────
# data_utils.py lives at the repo root, so processed/ is right next to it.
PROJECT_ROOT = Path(__file__).parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
MELODY_DIR = PROCESSED_DIR / "melodies_all"
DRUM_DIR = PROCESSED_DIR / "drums_all"
BASS_DIR = PROCESSED_DIR / "bass_all"

# General MIDI Drum Map (channel 9)
GM_DRUM_MAP = {
    35: "Acoustic Bass Drum", 36: "Kick",
    37: "Side Stick", 38: "Snare", 39: "Hand Clap", 40: "Electric Snare",
    41: "Low Floor Tom", 42: "Closed Hi-Hat", 43: "High Floor Tom",
    44: "Pedal Hi-Hat", 45: "Low Tom", 46: "Open Hi-Hat",
    47: "Low-Mid Tom", 48: "Hi-Mid Tom", 49: "Crash Cymbal 1",
    50: "High Tom", 51: "Ride Cymbal 1", 52: "Chinese Cymbal",
    53: "Ride Bell", 54: "Tambourine", 55: "Splash Cymbal",
    56: "Cowbell", 57: "Crash Cymbal 2", 58: "Vibraslap",
    59: "Ride Cymbal 2", 60: "High Bongo", 61: "Low Bongo",
    62: "Mute Hi Conga", 63: "Open Hi Conga", 64: "Low Conga",
    65: "High Timbale", 66: "Low Timbale", 67: "High Agogo",
    68: "Low Agogo", 69: "Cabasa", 70: "Maracas",
    71: "Short Whistle", 72: "Long Whistle", 73: "Short Guiro",
    74: "Long Guiro", 75: "Claves", 76: "High Wood Block",
    77: "Low Wood Block", 78: "Mute Cuica", 79: "Open Cuica",
    80: "Mute Triangle", 81: "Open Triangle",
}


# ─── Path resolution helper ────────────────────────────────────────────

def _resolve(path_value, default_dir):
    """Resolve a path that might be absolute (old) or just a filename (new).
    Always returns a path inside default_dir based on the filename, so this
    works regardless of whose machine generated the CSV."""
    name = Path(str(path_value)).name
    return default_dir / name


# ─── Metadata loaders ──────────────────────────────────────────────────

def load_metadata():
    """Load full song metadata (BPM, drums, melody info)."""
    return pd.read_csv(PROCESSED_DIR / "song_metadata.csv")


def load_melody_metadata():
    return pd.read_csv(PROCESSED_DIR / "melody_dataset.csv")


def load_drum_metadata():
    return pd.read_csv(PROCESSED_DIR / "drum_dataset.csv")


def load_bass_metadata():
    return pd.read_csv(PROCESSED_DIR / "bass_dataset.csv")


# ─── MIDI loaders ──────────────────────────────────────────────────────

def load_melody_midi(path):
    midi = pretty_midi.PrettyMIDI(str(path))
    return midi.instruments[0] if midi.instruments else None


def load_drum_midi(path):
    midi = pretty_midi.PrettyMIDI(str(path))
    return midi.instruments[0] if midi.instruments else None


def load_bass_midi(path):
    midi = pretty_midi.PrettyMIDI(str(path))
    return midi.instruments[0] if midi.instruments else None


# ─── Iterators (path-robust) ───────────────────────────────────────────

def iter_melodies():
    """Yield (metadata_row, pretty_midi.Instrument) for each melody."""
    df = load_melody_metadata()
    for _, row in df.iterrows():
        try:
            path = _resolve(row["melody_path"], MELODY_DIR)
            if not path.exists():
                continue
            inst = load_melody_midi(path)
            if inst is not None:
                yield row, inst
        except Exception:
            continue


def iter_drums():
    """Yield (metadata_row, pretty_midi.Instrument) for each drum track."""
    df = load_drum_metadata()
    for _, row in df.iterrows():
        try:
            path = _resolve(row["drum_path"], DRUM_DIR)
            if not path.exists():
                continue
            inst = load_drum_midi(path)
            if inst is not None:
                yield row, inst
        except Exception:
            continue


def iter_bass():
    """Yield (metadata_row, pretty_midi.Instrument) for each bass track."""
    df = load_bass_metadata()
    for _, row in df.iterrows():
        try:
            path = _resolve(row["bass_path"], BASS_DIR)
            if not path.exists():
                continue
            inst = load_bass_midi(path)
            if inst is not None:
                yield row, inst
        except Exception:
            continue


# ─── Quick Info ────────────────────────────────────────────────────────

def dataset_summary():
    """Print a summary of the dataset."""
    mel = load_melody_metadata()
    drm = load_drum_metadata()
    try:
        bas = load_bass_metadata()
        n_bass = len(bas)
    except FileNotFoundError:
        n_bass = 0

    print("═══════════════════════════════════════════")
    print("  Synth-pop Dataset")
    print("═══════════════════════════════════════════")
    print(f"  Melodies:    {len(mel)} songs")
    print(f"  Drum tracks: {len(drm)} songs")
    print(f"  Bass tracks: {n_bass} songs")
    print(f"  Artists:     {mel['artist'].nunique()}")
    print(f"  BPM range:   {mel['bpm'].min():.0f}-{mel['bpm'].max():.0f}")
    print(f"  Median BPM:  {mel['bpm'].median():.0f}")
    print("───────────────────────────────────────────")
    print("  Top melody instruments:")
    for instr, n in mel["instrument_name"].value_counts().head(5).items():
        print(f"    {instr}: {n}")
    print("═══════════════════════════════════════════")


if __name__ == "__main__":
    dataset_summary()
