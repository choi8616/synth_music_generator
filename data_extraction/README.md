# `data_extraction/` — Synth-pop Dataset Pipeline

**CSE 153 Final Project — Synth-pop Music Generation**

This folder contains the **data side** of the project: collecting, filtering, extracting, and validating a curated synth-pop MIDI dataset from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/). The modeling notebooks for the two generation tasks consume the output of this pipeline; they do not re-run any of the work here.

- **Task 1 — Symbolic, unconditioned generation:** generate synth-pop *melodies* from scratch.
- **Task 2 — Symbolic, conditioned generation:** generate *drum patterns* conditioned on BPM, style, and a short seed.

`baseline.ipynb` is the source of truth. `baseline.html` / `baseline.pdf` are rendered exports for reading without running the notebook.

---

## Where the data comes from

The pipeline builds on the **LMD "Clean MIDI" subset** — roughly 17,000 MIDI files organized into per-artist folders (`clean_midi/<Artist>/<Song>.mid`), drawn from the ~2,200 artists in that release.

**The raw MIDI corpus is *not* committed to this repo** (it is ~811 MB). Before running the notebook you must obtain the Clean MIDI subset separately and point the notebook at it. In the first setup cell, set:

```python
DATA_DIR = Path('/path/to/your/clean_midi')
```

Everything the notebook *produces* is written to `../processed/` at the repo root (also not committed, since it is large). The notebook regenerates all of it from `DATA_DIR`.

---

## What the notebook does

| Stage | Step | Key output |
|---|---|---|
| 1 | Locate the LMD Clean MIDI subset on disk | `DATA_DIR` |
| 2 | Survey artists in the corpus | — |
| 3 | Filter to synth-pop via a curated **~43-artist list** (80s classics, 90s electronic-pop, modern synth-pop) | ~384 candidate songs |
| 4 | Inspect one song in detail (track layout, piano rolls) to understand MIDI structure | — |
| 5 | BPM survey across the subset | ~374 valid-tempo songs |
| 6 | Drum-availability check | ~368 songs with drums |
| 7 | Persist metadata, then apply quality filters | `song_metadata.csv`, `usable_songs.csv` |
| 8 | **Task 1:** heuristic melody-track extraction | `melodies_all/`, `melody_dataset.csv` |
| 9 | **Task 2:** mechanical (channel-9) drum extraction | `drums_all/`, `drum_dataset.csv` |
| 10 | **Task 2 (extended):** bass-track extraction | `bass_all/`, `bass_dataset.csv` |
| 11 | Cross-match the three track types | `fully_paired_songs.csv` |
| 12 | Package loaders into a shared module | `../data_utils.py` |

### Cleaning / quality decisions worth knowing

- **Tempo:** BPM is read from each file's *stored* tempo events (`get_tempo_changes`) rather than `estimate_tempo()`, which frequently reports half- or double-time. The cleaned BPM distribution peaks at ~120, the textbook synth-pop tempo.
- **"Usable" filter:** a song is kept only if it has a drum track, has ≥ 50 drum notes (drops single-trigger artifacts), and sits at 70–160 BPM (drops outliers and remaining tempo glitches).
- **Melody extraction** is the hard part: LMD track names are inconsistent (`"Vocal"`, `"Lead"`, `"Synth String 4"`, or blank), so the melody track is chosen by a **score-based heuristic** (instrument program, pitch range, note density, monophony, name hints) rather than any single rule. The heuristic was iterated through three versions, each driven by listening to the extracted audio (e.g. v3 fixed picking a sparse vibraphone over the iconic tenor-sax hook in "Careless Whisper").
- **Drum extraction** is mechanical — MIDI channel 9 is reserved for percussion (`is_drum`), so no heuristic is needed; multi-track drum kits are merged into one track per song.
- **Bass extraction** uses bass program numbers (32–39) plus a low average-pitch signal, added later so the model could learn real basslines instead of faking them from melody roots.

---

## Outputs (written to `../processed/`)

| File / folder | Contents |
|---|---|
| `song_metadata.csv` | Per-song BPM + drum stats (source of truth) |
| `usable_songs.csv` | Songs passing the quality filters |
| `melodies_all/`, `melody_dataset.csv` | Extracted melody MIDIs + metadata |
| `drums_all/`, `drum_dataset.csv` | Extracted (merged) drum MIDIs + metadata |
| `bass_all/`, `bass_dataset.csv` | Extracted bass MIDIs + metadata |
| `fully_paired_songs.csv` | Songs that have melody **and** drums **and** bass |
| `melody_stats.png`, `drum_stats.png` | Summary plots used in the presentation |

---

## Headline results

- **335** melody MIDIs across **27** artists, BPM 70–159 (median **120**)
- **340** drum MIDIs (**97.6%** of usable songs had a usable drum track)
- Top extracted melody instruments: Lead 1 (square), Alto Sax, Acoustic Grand Piano, Lead 2 (sawtooth), Synth Choir — i.e. the pipeline independently rediscovers the canonical synth-pop palette, which is an implicit check that the heuristic works.

---

## Shared module: `data_utils.py`

The last cell writes a reusable module to the **repo root** (`../data_utils.py`). The modeling notebooks import from it directly so they never touch the extraction code:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))   # add repo root to path

from data_utils import dataset_summary, iter_melodies, iter_drums, iter_bass

dataset_summary()
for meta, instrument in iter_melodies():
    ...  # meta is a metadata row; instrument is a pretty_midi.Instrument
```

> Note: some markdown cells refer to this as `src/data_utils.py`, but the file is actually written to the repo root. Import it as `data_utils`, not `src.data_utils`.

---

## Running it

Requirements:

```bash
pip install pretty_midi numpy pandas matplotlib tqdm jupyter
```

Then:

1. Obtain the LMD Clean MIDI subset and set `DATA_DIR` in the first setup cell.
2. Open `baseline.ipynb` **from inside this `data_extraction/` folder** (the notebook uses `Path.cwd().parent` to find the repo root and write to `../processed/`).
3. Run all cells top to bottom. A full run re-extracts all three track types and regenerates everything in `../processed/`.

A couple of caveats from the notebook itself: the inline audio uses `pretty_midi.synthesize()`, a basic sine synth that renders **drums** thinly — open the drum `.mid` files in a DAW for an honest listen; the MIDI data is correct. Melody and bass render fine inline.
