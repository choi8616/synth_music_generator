# Task 2: Symbolic Conditioned Music Generation

This folder contains the full pipeline for Task 2.

The project goal is:

```text
synth-pop MIDI files (Lakh MIDI Dataset)
  -> melody track + drum track extraction
  -> 16-step grid representation (4-bar segments)
  -> rule-based baseline (bass + drums)
  -> ConditionalDrumLSTM (melody + BPM -> drum groove)
  -> generated MIDI (melody + bass + drums)
```

## Big Picture

The central idea is **conditional generation**:

```text
Given a melody sequence and BPM, predict the drum groove that fits it.
```

For example, a 4-bar melody becomes a fixed-length integer sequence:

```text
[0, 0, 69, 0, 71, 0, 72, 0, ...]   (length 64, 0 = rest, 1~128 = MIDI pitch + 1)
```

And the target drum groove becomes a binary matrix:

```text
step:  0  1  2  3  4  5  6  7  8 ...
kick:  1  0  0  0  0  0  0  0  1 ...
snare: 0  0  0  0  1  0  0  0  0 ...
hat:   1  0  1  0  1  0  1  0  1 ...
```

### Why 16-step grid?

Most 80s synth-pop follows a strict 4/4 time signature with 16th-note quantization.
One bar = 16 steps, 4 bars = 64 steps total. This covers one full musical phrase.

```text
1 e & a  2 e & a  3 e & a  4 e & a
| . . .  | . . .  | . . .  | . . .   (16 steps per bar)
```

### Two models

**Rule-based baseline**: handcrafted patterns, no learning required.

```text
melody -> estimate root pitch -> fixed bass pattern (root/fifth/octave)
drums  -> fixed pattern: kick on 1/3, snare on 2/4, hi-hat every 8th note
```

**ConditionalDrumLSTM**: learns from data.

```text
melody embedding + BPM scalar -> LSTM hidden states -> drum logits (3 channels)
p(kick=1 | melody, BPM),  p(snare=1 | melody, BPM),  p(hat=1 | melody, BPM)
```

The LSTM learns that different melodic phrases call for different rhythmic accompaniments,
while the rule-based baseline always produces the same pattern regardless of the melody.

## Folder Layout

```text
task2/
  processed/
    bpm_data.csv              # BPM per song
    melody_dataset.csv        # metadata for extracted melody tracks
    drum_dataset.csv          # metadata for extracted drum tracks
    usable_songs.csv          # songs with both melody and drums
    song_metadata.csv         # full song metadata
    melodies_all/             # extracted melody MIDI files (one per song)
    drums_all/                # extracted drum MIDI files (one per song)
    task2_drum_dataset_4bar.npz   # final training dataset
  src/
    __init__.py
    data_utils.py             # MIDI loading, track extraction, dataset summary
  task2222.ipynb              # main notebook (EDA -> preprocessing -> training -> generation)
  README.md
```

## Step 1: Understand the Dataset

### Dataset Context

The dataset is built on the **Lakh MIDI Dataset (LMD)** (Raffel, 2016), one of the largest
publicly available collections of MIDI files, containing approximately 176,581 unique MIDI files
matched to entries in the Million Song Dataset. From this corpus, synth-pop tracks were filtered
and curated by extracting songs from representative 80s synth-pop artists including
Depeche Mode, Culture Club, ABC, Bronski Beat, and others.

For each song, two tracks were automatically extracted:
- **Melody track**: selected by scoring instruments on pitch range (C4–C6), note density,
  monophony ratio, and instrument program number (preferring lead synths, brass, and winds).
- **Drum track**: identified via MIDI channel 9 (`is_drum=True`) and merged if multiple
  drum tracks were present.

Only songs with both a valid melody track and a drum track were retained (`usable_songs.csv`).

### Dataset Statistics (expected ranges)

```text
Total matched songs:  ~70–90
4-bar segments:       4,882
BPM range:            ~90–130 BPM (typical 80s synth-pop tempo)
Segment duration:     ~8 seconds per segment (at 120 BPM)
Drum classes:         kick, snare, hi-hat (3 classes from GM drum map)
```

### Exploratory Analysis (to verify in the notebook)

The following plots should be generated from the metadata before training:

```python
import matplotlib.pyplot as plt

# 1. BPM distribution
plt.hist(bpm_df["bpm"], bins=20)
plt.title("BPM Distribution"); plt.xlabel("BPM"); plt.show()

# 2. Melody pitch distribution
plt.hist(melody_df["avg_pitch"], bins=20)
plt.title("Average Melody Pitch"); plt.xlabel("MIDI Pitch"); plt.show()

# 3. Drum hit density per class
print("kick  hit rate:", drums_train[:, :, 0].mean().round(3))
print("snare hit rate:", drums_train[:, :, 1].mean().round(3))
print("hat   hit rate:", drums_train[:, :, 2].mean().round(3))
```

Melody and drum tracks have been pre-extracted and saved to `processed/melodies_all/` and `processed/drums_all/`.

Load the metadata in the notebook:

```python
import pandas as pd

bpm_df    = pd.read_csv("processed/bpm_data.csv")
melody_df = pd.read_csv("processed/melody_dataset.csv")
drum_df   = pd.read_csv("processed/drum_dataset.csv")
```

Key columns in `melody_dataset.csv`:

```text
artist, song, melody_path, instrument_program, instrument_name, n_notes, avg_pitch, bpm
```

Key columns in `drum_dataset.csv`:

```text
artist, song, drum_path, n_drum_tracks_merged, n_notes, n_unique_drums, bpm, duration
```

## Step 2: Build the 16-step Grid Dataset

The notebook converts each matched melody+drum MIDI pair into 4-bar segments:

```python
segments = midi_pair_to_segments(
    melody_path,
    drum_path,
    min_melody_notes=8,
    min_drum_hits=8
)
```

Each segment is a dictionary:

```python
{
    "melody": np.array of shape (64,),      # pitch tokens, 0=rest
    "drums":  np.array of shape (64, 3),    # [kick, snare, hat] binary
    "bpm":    float,
    "start_time": float
}
```

The full dataset is saved as a compressed numpy archive:

```python
np.savez(
    "processed/task2_drum_dataset_4bar.npz",
    melodies=all_melodies,   # shape: (N, 64)
    drums=all_drums,         # shape: (N, 64, 3)
    bpms=all_bpms,           # shape: (N,)
    files=...,
    start_times=...
)
```

Expected dataset size after preprocessing: approximately **4,882 segments**.

### Drum class mapping

Only three drum classes are used, merged from General MIDI pitch numbers:

```text
kick:   MIDI 35, 36
snare:  MIDI 38, 40
hi-hat: MIDI 42, 44, 46
```

## Step 3: Rule-Based Baseline

Before training, a rule-based generator establishes a lower-bound baseline:

```python
bass_grid, drum_grid = create_rule_based_synthpop_midi(
    melody_grid=melody_grid,
    bpm=120,
    output_path="rule_based_synthpop.mid"
)
```

The rule-based system:

1. Estimates the root pitch by finding the most common pitch class in the melody
2. Builds a repeating bass pattern: `root, -, root, -, fifth, -, root, -, octave, ...`
3. Applies a fixed drum pattern: kick on beats 1 and 3, snare on 2 and 4, hi-hat every 8th note

This baseline produces the same output regardless of the melody's internal structure.
It is used as a comparison point for the LSTM model.

### Rule-based vs LSTM: Advantages and Disadvantages

| | Rule-based | ConditionalDrumLSTM |
|---|---|---|
| Training required | No | Yes (30 epochs) |
| Melody-aware | Partial (root pitch only) | Yes (full sequence) |
| Output diversity | None (always identical) | High (stochastic mode) |
| Beat regularity | Perfect (hardcoded) | Lower (learns from data) |
| Hit rate accuracy (MAE) | 0.0406 | **0.0091** (4.5x better) |
| Handles syncopation | No | Yes |
| Interpretable | Yes | No (black box) |

The rule-based model guarantees rhythmic regularity but cannot adapt to the melody's
phrasing or dynamics. The LSTM sacrifices some regularity for a much more accurate
hit rate distribution and the ability to produce varied patterns per melody.

### Why LSTM over Transformer?

An LSTM was chosen over a Transformer for the following reasons:

1. **Sequence length**: 64-step sequences are short enough that LSTMs do not suffer
   from the vanishing-gradient issues that typically motivate Transformer use.
2. **Data size**: With ~4,882 training segments, a Transformer's self-attention mechanism
   would be prone to overfitting. LSTMs generalize better on smaller datasets.
3. **Simplicity**: The LSTM architecture is easier to interpret and debug, making it
   a more suitable baseline for a course project.

A Transformer encoder-decoder with cross-attention between melody and drum sequences
would be a natural next step for larger datasets.

## Step 4: Handle Class Imbalance

Before training, compute `pos_weight` from `drums_train` to correct for drum class imbalance.
This must be done **after** `train_test_split`, using only the training split:

```python
from sklearn.model_selection import train_test_split

(melody_train, melody_val,
 bpm_train,    bpm_val,
 drums_train,  drums_val) = train_test_split(
    X_melody, X_bpm, Y_drums,
    test_size=0.1, random_state=42
)

# Compute pos_weight from drums_train only (not Y_drums)
positive_counts = drums_train.sum(axis=(0, 1))   # shape: (3,)
total_counts    = drums_train.shape[0] * drums_train.shape[1]
negative_counts = total_counts - positive_counts

pos_weight = np.sqrt(negative_counts / (positive_counts + 1e-8))
```

Expected `pos_weight` values (approximate):

```text
kick:   ~1.83   (fires ~23% of steps)
snare:  ~2.60   (fires ~13% of steps — rarest)
hi-hat: ~1.16   (fires ~42% of steps — most common)
```

`np.sqrt` is used to moderate the weighting: the raw ratio would over-penalize hi-hat misses
and cause the model to over-generate kick and snare in the opposite direction.

## Step 5: Train the ConditionalDrumLSTM

```python
model = ConditionalDrumLSTM(
    melody_vocab_size=129,   # 0=rest, 1~128=MIDI pitch+1
    melody_emb_dim=32,
    hidden_dim=128,
    num_layers=2,
    drum_dim=3,
    dropout=0.2
)

model = train_conditional_drum_lstm(
    model,
    train_loader,
    pos_weight=pos_weight,
    num_epochs=30,
    lr=0.001
)
```

### Architecture

```text
melody (64,) -> Embedding (64, 32)
BPM scalar   -> repeated (64, 1)
concatenated -> (64, 33)
LSTM         -> hidden states (64, 128)
Linear       -> drum logits (64, 3)   <- one logit per drum per step
BCEWithLogitsLoss(pos_weight=pos_weight)
```

BPM is normalized before use:

```python
bpm_norm = (bpm - 120.0) / 40.0
```

Loss uses `BCEWithLogitsLoss` with `pos_weight`, which applies per-class weighting directly
to the loss function. This trains the model to pay more attention to rare drum hits (snare)
rather than just predicting silence everywhere.

## Step 6: Generate Drums — Two Modes

### Mode A: Deterministic (threshold-based)

Same melody always produces the same drum pattern. Uses per-drum thresholds
tuned to match the hit rate distribution of the training data:

```python
drum_pred = generate_drums(
    model,
    melody_grid,
    bpm,
    thresholds=(0.45, 0.38, 0.50)   # (kick, snare, hi-hat)
)
```

Threshold values were chosen by comparing the model's predicted hit rates against
the actual training data hit rates for each drum class.

### Mode B: Stochastic (temperature-based)

Same melody produces a different drum pattern each time. Uses temperature scaling
on the raw logits before sigmoid, then samples from the resulting Bernoulli distribution:

```python
drum_pred = generate_drums_stochastic(
    model,
    melody_grid,
    bpm,
    temperature=1.0    # > 1.0 = more random, < 1.0 = more conservative
)
```

Temperature is applied **before** sigmoid (logit scaling), not after:

```text
logits / temperature -> sigmoid -> Bernoulli sample
```

This is the standard temperature sampling approach used in language models.
At `temperature=1.0`, the model outputs its learned probabilities unchanged.

### Which mode to use?

```text
Deterministic: reproducible evaluation, comparison against baseline, sanity checks
Stochastic:    generating multiple variations from the same melody, final output
```

## Step 7: Reconstruct MIDI

Both generation modes output a `drum_pred` array of shape `(64, 3)`.
Pass this to `create_lstm_midi` along with the original melody and BPM:

```python
create_lstm_midi(
    melody_grid=sample_melody,
    drum_pred=drum_pred,
    bpm=sample_bpm,
    output_path="lstm_synthpop.mid"
)
```

The output MIDI contains three tracks:

```text
Track 1: Melody Synth Lead  (program 80)
Track 2: Rule-Based Synth Bass (program 38)  <- still rule-based
Track 3: LSTM Drums          (channel 9, is_drum=True)
```

Note: the bass track remains rule-based in the current version.
The LSTM only generates the drum groove.

## Evaluation: What Makes a Good Drum Groove?

A generated drum groove is considered good if it satisfies the following properties:

1. **Hit rate accuracy**: The frequency of kick, snare, and hi-hat hits should match
   the distribution found in real synth-pop drum tracks.
   - Evaluated with Mean Absolute Error (MAE) against training data hit rates.

2. **No collapse**: Every segment should contain at least one kick and one snare.
   A model that only predicts hi-hat on every step has collapsed.
   - Evaluated by checking `kick.sum() > 0 and snare.sum() > 0` per segment.

3. **Diversity**: Given the same melody, the model should be able to produce
   different drum patterns (relevant for stochastic mode).
   - Evaluated by running 10 generations from the same melody and measuring
     standard deviation of total hit counts per drum.

4. **Beat alignment**: Kicks and snares should tend to land near expected beat
   positions (beats 1/3 for kick, beats 2/4 for snare), reflecting 80s synth-pop
   rhythmic conventions.
   - Note: lower alignment than rule-based does not necessarily indicate failure;
     real drums also contain syncopation and ghost notes.

### Evaluation Results Summary

```text
Method                  MAE     kick/bar  snare/bar  hat/bar  diversity
----------------------------------------------------------------------
Real data (train)       —       3.68      2.06       6.79     —
Rule-based baseline     0.0406  3.00      2.00       8.00     none (std=0)
LSTM (deterministic)    0.0091  3.82      2.17       6.59     fixed
LSTM (stochastic t=1)   0.0680  5.07      3.12       7.62     kick std=3.6
```

The LSTM deterministic mode achieves the lowest MAE (0.0091), meaning its hit rate
distribution is 4.5x closer to real data than the rule-based baseline.

### Audio Comparison

To compare the outputs directly, play them side by side in the notebook:

```python
import IPython.display as ipd
print("Rule-based:")
ipd.display(ipd.Audio("rule_based_synthpop.mid"))
print("LSTM (deterministic):")
ipd.display(ipd.Audio("lstm_deterministic.mid"))
print("LSTM (stochastic):")
ipd.display(ipd.Audio("lstm_stochastic.mid"))
```

## Related Work

### Dataset

**Lakh MIDI Dataset (LMD)** (Raffel, C., 2016. *Learning-Based Methods for Comparing
Sequences, with Applications to Audio-to-MIDI Alignment and Matching*. PhD thesis,
Columbia University.)
The LMD contains ~176K MIDI files matched to the Million Song Dataset. It has been
widely used for symbolic music generation, transcription, and style transfer tasks.
This project uses a synth-pop filtered subset of the LMD.

### Drum Generation

**Groove MIDI Dataset** (Gillick, J. et al., 2019. *Learning to Groove with Inverse
Kinematics*. ICML 2019.)
A dataset of 13.6 hours of drum performances captured from a Roland electronic drum kit.
Used to train GrooVAE, a variational autoencoder that generates humanized drum grooves.
Unlike this project, GrooVAE operates on continuous timing and velocity rather than
a discrete 16-step grid.

**GrooVAE** (Google Magenta, 2019.)
A VAE-based model conditioned on a "tap" rhythm input that generates expressive drum
grooves. Related to this project in that it performs conditioned drum generation, though
the conditioning signal (tap rhythm vs. melody) and architecture (VAE vs. LSTM) differ.

### Conditioned Symbolic Generation

**Pop Music Transformer / REMI** (Huang, Y.S. and Yang, Y.H., 2020. *Pop Music
Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Music*.
ACM MM 2020.)
Introduced the REMI token representation (used in Task 1 of this project) for expressive
piano music generation. Demonstrates that richer token representations capture musical
structure more faithfully than raw MIDI events.

**Music Transformer** (Huang, C.A. et al., 2018. *Music Transformer: Generating Music
with Long-Term Structure*. ICLR 2019.)
A Transformer-based model for symbolic music generation with relative attention,
enabling long-range dependency modeling. Represents the direction this project's LSTM
model could be extended toward with more data.

### How This Project Relates to Prior Work

Prior drum generation work (GrooVAE, Magenta Drumify) typically:
- Conditions on a tap or rhythm input, not a melody
- Uses continuous representations (timing offsets, velocity) for expressiveness
- Operates on longer sequences or full songs

This project differs by:
- Conditioning directly on the **melody pitch sequence**, framing drum generation as
  a sequence-to-sequence ML problem
- Using a simple **discrete 16-step binary grid**, trading expressiveness for simplicity
- Targeting a specific genre (80s synth-pop) with a small curated dataset

## Key Parameters Reference

| Parameter | Where | Effect |
|---|---|---|
| `test_size=0.1` | train_test_split | 10% validation split |
| `pos_weight` | BCEWithLogitsLoss | corrects class imbalance at training time |
| `num_epochs=30` | training | loss was still decreasing at 20; 30 improves convergence |
| `thresholds=(0.45, 0.38, 0.50)` | generate_drums | per-drum decision boundary |
| `temperature=1.0` | generate_drums_stochastic | controls randomness of stochastic output |
| `STEPS_PER_BAR=16` | grid representation | 16th-note quantization |
| `BARS_PER_SEGMENT=4` | grid representation | 4-bar (64-step) segments |

## What You Should Understand First

`task2222.ipynb` is the single entry point for the full pipeline. The key questions it answers:

1. What does the synth-pop dataset look like? (BPM distribution, melody pitch range, drum density)
2. How is a MIDI file turned into a fixed-length grid that a neural network can consume?
3. Why does a naive LSTM collapse to predicting only hi-hat, and how does `pos_weight` fix it?
4. How do deterministic and stochastic generation differ in practice?
5. Does the LSTM produce more musically appropriate drums than the rule-based baseline?

Once the preprocessing and baseline are understood, the LSTM is just a learned replacement
for the rule-based drum generator — conditioned on the same melody input.
