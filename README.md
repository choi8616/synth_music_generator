# 🎹 Synth-Pop Symbolic Music Generation


Two symbolic-music generators trained on a self-curated **80s synth-pop** MIDI dataset built from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/). We tackle two of the four project tasks, both in the symbolic domain:

- **Task 1 — Unconditioned generation:** learn `p(melody, drums)` and generate new 4-bar synth-pop sketches **from scratch**.
- **Task 2 — Conditioned generation:** given a **melody + tempo**, generate a matching **bassline and drum groove** (melody-conditioned accompaniment).

Both tasks share a **4-bar / 16-steps-per-bar grid** representation. The complete write-up — data pipeline, both models, evaluation, plots, and audio — lives in [`workbook.ipynb`](workbook.ipynb) (open the rendered [`workbook.html`](workbook.html) to read it without running anything).

---

## Generated samples

| File | Task | What it is |
|---|---|---|
| `symbolic_unconditioned.mid` | Task 1 | LSTM-generated melody + drums, no input prompt |
| `symbolic_conditioned.mid` | Task 2 | Melody-conditioned bass + drums (LSTM, stochastic decoding) |

---

## Repository structure

```
synth_music_generator/
├── workbook.ipynb            # Combined write-up: data + Task 1 + Task 2 (the deliverable)
├── workbook.html             # Same, rendered for reading in a browser
├── data_utils.py             # Shared dataset loaders (iter_melodies / iter_drums / iter_bass)
│
├── data_extraction/          # Part 1 — dataset collection, extraction & cleaning
│   └── baseline.ipynb
├── task1/                    # Task 1 — symbolic unconditioned generation
│   ├── task1_final.ipynb
│   ├── src/                  #   model + training code
│   └── outputs_final/        #   generated MIDI
├── task2/                    # Task 2 — symbolic conditioned generation
│   ├── task2.ipynb
│   └── src/
│
├── processed/                # Extracted melody/drum/bass MIDIs + metadata CSVs (generated)
├── symbolic_unconditioned.mid
└── symbolic_conditioned.mid
```

> **Large files are not committed.** The raw MIDI corpus (~811 MB) and the regenerated `processed/` data are excluded; they are rebuilt from the source dataset (see below).

---

## The dataset (Part 1)

Curated from the **LMD "Clean MIDI" subset** (~17,000 MIDI files organized by artist). Because LMD has no reliable genre labels, we filtered with a hand-built list of **~43 synth-pop artists** across three eras (80s classics, 90s electronic-pop, modern synth-pop), giving ~384 candidate songs, then cleaned and separated them into melody / drum / bass tracks.

| Stage | Output |
|---|---|
| Filter to synth-pop artists | ~384 candidate songs |
| BPM survey (from stored tempo events) + drum check | peaks at **120 BPM**; 97.6% have drums |
| Quality filter (has drums, ≥50 drum notes, BPM 70–160) | usable set |
| Melody extraction (score-based heuristic, 3 iterations) | **335** melody MIDIs |
| Drum extraction (MIDI channel 9, merged kits) | **340** drum MIDIs |
| Bass extraction (bass program + low pitch) | **334** bass MIDIs |

**Final dataset:** 335 melodies / 340 drums / 334 basslines across **27 artists**, BPM 70–159 (median 120). The most common extracted lead instruments are square/sawtooth synth leads, alto sax, and synth choir — i.e. the pipeline independently recovers the canonical synth-pop palette. All loaders are packaged in `data_utils.py`.

*(Two implementation notes worth knowing: tempo is read from each file's stored tempo events rather than `pretty_midi.estimate_tempo()`, which returns half/double-time; and melody extraction is feature-based, not name-based, because LMD track names are inconsistent.)*

---

## Task 1 — Symbolic Unconditioned Generation (melody + drums)

- **Representation:** 4 bars × 16 steps = 64-step grid. Each step encodes a melody state (rest / hold / pitch) and a 3-bit drum mask (kick / snare / hat), combined into a single stream of ~375 event tokens.
- **Models:** an **order-5 Markov chain** baseline vs. an **LSTM event language model** (embedding → LSTM → softmax over the event vocabulary).
- **Evaluation:** validation **perplexity** (the LSTM improves on the Markov baseline) plus a **pitch-class distribution** check against the real corpus (a tonal-consistency proxy that likelihood alone misses), and a sampling-**temperature** ablation.

---

## Task 2 — Symbolic Conditioned Generation (melody + BPM → bass + drums)

- **Representation:** input is the 64-step melody plus a normalized BPM feature; outputs are a bass token per step and a 3-class drum activation (kick / snare / hat) per step.
- **Models:** a hand-written **rule-based baseline** (kick on 1/3, snare on 2/4, hi-hat on 8ths; bass from the melody's most common pitch class) vs. a **conditional LSTM** with two heads (bass + drums) decoded in **deterministic** and **stochastic** modes.
- **Evaluation:** per-bar hit/activation rates vs. real data, **MAE from the real distribution**, output **diversity** (std across runs), and a loss-weight ablation.

**Results (lower MAE = better):**

| Method | Bass MAE | Drum MAE | Total MAE | |
|---|---|---|---|---|
| Rule-based | 1.43 | 0.60 | 2.03 | |
| LSTM Deterministic | 0.77 | 0.06 | 0.83 | |
| **LSTM Stochastic** | **0.26** | 0.24 | **0.50** | ✅ selected |

The stochastic LSTM matches the real synth-pop distribution best and adds per-melody variety (the rule-based baseline has zero diversity by construction); `symbolic_conditioned.mid` is generated with it.

---

## Reproducing

```bash
# 1. Install dependencies
pip install pretty_midi numpy pandas matplotlib tqdm torch scikit-learn jupyter

# 2. Obtain the LMD "Clean MIDI" subset and point the data-extraction notebook at it
#    (set DATA_DIR at the top of data_extraction/baseline.ipynb)

# 3. Build the dataset, then run the models
jupyter notebook
```

Recommended order: run `data_extraction/baseline.ipynb` first to populate `processed/`, then the Task 1 and Task 2 notebooks (or the combined `workbook.ipynb`). For a clean full run, use **Kernel → Restart & Run All** so every cell executes in order. Data-directory paths reflect our development layout — adjust them at the top of each notebook if your checkout differs.

---

## Related work

- **Lakh MIDI Dataset** — Raffel (2016) · the source corpus.
- **GrooVAE** / **Groove MIDI Dataset** — Magenta; Gillick et al. (2019) · learned drum grooves (drums only, conditioned on tap rhythm).
- **Music Transformer** / **Pop Music Transformer (REMI)** — Huang et al. (2018); Huang & Yang (2020) · Transformer models of long-range structure.

Unlike this prior work, we condition on the **full melody pitch sequence** (not a tap rhythm or chord label), generate **bass and drums jointly** from one model, and use a compact discrete grid suited to our small, genre-specific corpus.

---

## Team

| Area | Member |
|---|---|
| Data collection, extraction & cleaning (Part 1) | _[name]_ |
| Task 1 — unconditioned generation | _[name]_ |
| Task 2 — conditioned generation | _[name]_ |

## Acknowledgments

Built with [pretty_midi](https://github.com/craffel/pretty-midi) and [PyTorch](https://pytorch.org/), on the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/).
