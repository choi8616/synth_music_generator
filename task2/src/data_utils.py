"""
Synth-pop Music Generation — Data Utilities
============================================

Shared functions for loading and processing the curated synth-pop MIDI dataset.

Usage from a teammate's notebook:
    import sys
    sys.path.insert(0, '/Users/donghyunhahn/Desktop/spring 2026/cse 153/Assignment 2')
    from src.data_utils import load_melody_dataset, load_drum_dataset
    
    melodies = load_melody_dataset()
    drums = load_drum_dataset()
"""

import pretty_midi
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'processed'
MELODY_DIR = PROCESSED_DIR / 'melodies_all'
DRUM_DIR = PROCESSED_DIR / 'drums_all'

# General MIDI Drum Map (channel 9)
GM_DRUM_MAP = {
    35: 'Acoustic Bass Drum', 36: 'Kick',
    37: 'Side Stick', 38: 'Snare', 39: 'Hand Clap', 40: 'Electric Snare',
    41: 'Low Floor Tom', 42: 'Closed Hi-Hat', 43: 'High Floor Tom',
    44: 'Pedal Hi-Hat', 45: 'Low Tom', 46: 'Open Hi-Hat',
    47: 'Low-Mid Tom', 48: 'Hi-Mid Tom', 49: 'Crash Cymbal 1',
    50: 'High Tom', 51: 'Ride Cymbal 1', 52: 'Chinese Cymbal',
    53: 'Ride Bell', 54: 'Tambourine', 55: 'Splash Cymbal',
    56: 'Cowbell', 57: 'Crash Cymbal 2', 58: 'Vibraslap',
    59: 'Ride Cymbal 2', 60: 'High Bongo', 61: 'Low Bongo',
    62: 'Mute Hi Conga', 63: 'Open Hi Conga', 64: 'Low Conga',
    65: 'High Timbale', 66: 'Low Timbale', 67: 'High Agogo',
    68: 'Low Agogo', 69: 'Cabasa', 70: 'Maracas',
    71: 'Short Whistle', 72: 'Long Whistle', 73: 'Short Guiro',
    74: 'Long Guiro', 75: 'Claves', 76: 'High Wood Block',
    77: 'Low Wood Block', 78: 'Mute Cuica', 79: 'Open Cuica',
    80: 'Mute Triangle', 81: 'Open Triangle',
}


# ─── Dataset Loaders ───────────────────────────────────────────────────

def load_metadata():
    """Load full song metadata (BPM, drums, melody info)."""
    return pd.read_csv(PROCESSED_DIR / 'song_metadata.csv')


def load_melody_metadata():
    """Load metadata for extracted melodies."""
    return pd.read_csv(PROCESSED_DIR / 'melody_dataset.csv')


def load_drum_metadata():
    """Load metadata for extracted drum tracks."""
    return pd.read_csv(PROCESSED_DIR / 'drum_dataset.csv')


def load_melody_midi(melody_path):
    """Load a single melody MIDI and return the Instrument object."""
    midi = pretty_midi.PrettyMIDI(str(melody_path))
    return midi.instruments[0] if midi.instruments else None


def load_drum_midi(drum_path):
    """Load a single drum MIDI and return the Instrument object."""
    midi = pretty_midi.PrettyMIDI(str(drum_path))
    return midi.instruments[0] if midi.instruments else None


def iter_melodies():
    """Yield (metadata_row, pretty_midi.Instrument) for each melody."""
    df = load_melody_metadata()
    for _, row in df.iterrows():
        try:
            inst = load_melody_midi(row['melody_path'])
            if inst is not None:
                yield row, inst
        except Exception:
            continue


def iter_drums():
    """Yield (metadata_row, pretty_midi.Instrument) for each drum track."""
    df = load_drum_metadata()
    for _, row in df.iterrows():
        try:
            inst = load_drum_midi(row['drum_path'])
            if inst is not None:
                yield row, inst
        except Exception:
            continue


# ─── Track Extraction (for re-extracting from raw LMD if needed) ───────

def find_melody_track(midi):
    """
    Find the most melody-like track in a multi-track MIDI.
    Used during data preparation; teammates won't usually call this.
    """
    duration = midi.get_end_time()
    if duration < 30:
        return None
    
    expected_min_notes = int(duration * 0.3)
    
    STRONG_LEAD = set(
        list(range(80, 88))
        + list(range(56, 64))   # Brass
        + list(range(64, 72))   # Reeds
        + [52, 53, 54]          # Choir / Voice
        + [73, 74, 75]          # Pipes/flutes
    )
    WEAK_MELODY = set(list(range(0, 8)) + list(range(24, 31)) + list(range(40, 48)))
    
    candidates = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        n_notes = len(inst.notes)
        if n_notes < max(expected_min_notes, 50):
            continue
        
        pitches = [n.pitch for n in inst.notes]
        avg_pitch = np.mean(pitches)
        if not (52 <= avg_pitch <= 84):
            continue
        
        sorted_notes = sorted(inst.notes, key=lambda n: n.start)
        overlaps = sum(
            1 for i in range(len(sorted_notes) - 1)
            if sorted_notes[i + 1].start < sorted_notes[i].end - 0.05
        )
        mono_score = 1 - (overlaps / max(1, len(sorted_notes)))
        if mono_score < 0.4:
            continue
        
        score = 0
        if inst.program in STRONG_LEAD:
            score += 0.6
        elif inst.program in WEAK_MELODY:
            score += 0.15
        
        name_lower = (inst.name or '').lower()
        if any(k in name_lower for k in ['melody', 'vocal', 'voice', 'sing', 'main']):
            score += 0.5
        elif any(k in name_lower for k in ['lead', 'solo', 'sax']):
            score += 0.3
        
        density = n_notes / duration
        if 0.5 <= density <= 3.5:
            score += 0.2
        elif density > 3.5:
            score -= 0.1
        
        score += mono_score * 0.3
        
        pitch_range = max(pitches) - min(pitches)
        if 7 <= pitch_range <= 30:
            score += 0.15
        
        candidates.append((inst, score))
    
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def find_drum_track(midi):
    """Find and merge all drum tracks in a MIDI."""
    drum_tracks = [inst for inst in midi.instruments if inst.is_drum]
    if not drum_tracks:
        return None
    
    merged = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for dt in drum_tracks:
        merged.notes.extend(dt.notes)
    merged.notes.sort(key=lambda n: n.start)
    return merged


# ─── Quick Info ────────────────────────────────────────────────────────

def dataset_summary():
    """Print a summary of the dataset."""
    mel = load_melody_metadata()
    drm = load_drum_metadata()
    
    print(f"═══════════════════════════════════════════")
    print(f"  Synth-pop Dataset")
    print(f"═══════════════════════════════════════════")
    print(f"  Melodies: {len(mel)} songs")
    print(f"  Drum tracks: {len(drm)} songs")
    print(f"  Artists: {mel['artist'].nunique()}")
    print(f"  BPM range: {mel['bpm'].min():.0f}-{mel['bpm'].max():.0f}")
    print(f"  Median BPM: {mel['bpm'].median():.0f}")
    print(f"───────────────────────────────────────────")
    print(f"  Top instruments in melodies:")
    for instr, n in mel['instrument_name'].value_counts().head(5).items():
        print(f"    {instr}: {n}")
    print(f"═══════════════════════════════════════════")


if __name__ == '__main__':
    dataset_summary()