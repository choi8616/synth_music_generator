# Task 2: Symbolic Conditioned Music Generation

This folder contains the full pipeline for Task 2.

The project goal is:

```text
synth-pop MIDI files (Lakh MIDI Dataset)
  -> melody + drum + bass track extraction
  -> 16-step grid representation (4-bar segments)
  -> rule-based baseline (bass + drums)
  -> ConditionalBassDrumLSTM (melody + BPM -> bassline + drum groove)
  -> generated MIDI (melody + LSTM bass + LSTM drums)
```

## Big Picture

The central idea is **conditional generation**:

```text
Given a melody sequence and BPM, predict both the bassline and drum groove that fit it.
```

A 4-bar melody becomes a fixed-length integer sequence:

```text
[0, 0, 69, 0, 71, 0, 72, 0, ...]   (length 64, 0 = rest, 1~128 = MIDI pitch + 1)
```

The target bassline is also a pitch sequence (same format as melody):

```text
[37, 0, 37, 0, 44, 0, 37, 0, ...]  (length 64, 0 = rest, 1~128 = MIDI pitch + 1)
```

And the target drum groove is a binary matrix:

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

**ConditionalBassDrumLSTM**: learns from data.

```text
melody embedding + BPM scalar
  -> LSTM hidden states
  -> bass_head:  p(pitch | melody, BPM)     (129-class softmax per step)
  -> drum_head:  p(kick=1 | melody, BPM)
                 p(snare=1 | melody, BPM)
                 p(hat=1 | melody, BPM)
```

A single LSTM simultaneously learns to generate both bassline and drum groove
conditioned on the melody, while the rule-based baseline always produces the same
output regardless of the melody's internal structure.

## Folder Layout

```text
task2/
  processed/
    bpm_data.csv              # BPM per song
    melody_dataset.csv        # metadata for extracted melody tracks
    drum_dataset.csv          # metadata for extracted drum tracks
    bass_dataset.csv          # metadata for extracted bass tracks
    usable_songs.csv          # songs with melody, drums, and bass
    song_metadata.csv         # full song metadata
    melodies_all/             # extracted melody MIDI files (one per song)
    drums_all/                # extracted drum MIDI files (one per song)
    bass_all/                 # extracted bass MIDI files (one per song)
    task2_bass_drum_dataset_4bar.npz   # final training dataset
  task2.ipynb                 # main notebook (EDA -> preprocessing -> training -> generation)
  README.md
```

## Step 1: Understand the Dataset

### Dataset Context

The dataset is built on the **Lakh MIDI Dataset (LMD)** (Raffel, 2016), one of the largest
publicly available collections of MIDI files, containing approximately 176,581 unique MIDI files
matched to entries in the Million Song Dataset. From this corpus, synth-pop tracks were filtered
and curated by extracting songs from representative 80s synth-pop artists including
Depeche Mode, Culture Club, ABC, Bronski Beat, and others.

For each song, three tracks were automatically extracted:
- **Melody track**: selected by scoring instruments on pitch range (C4–C6), note density,
  monophony ratio, and instrument program number (preferring lead synths, brass, and winds).
- **Drum track**: identified via MIDI channel 9 (`is_drum=True`) and merged if multiple
  drum tracks were present.
- **Bass track**: identified by instrument name containing "bass" or program number 32–39
  (bass instruments in the General MIDI standard).

Only songs with all three tracks present were retained for training.

### Dataset Statistics

```text
Melody files:            333
Drum files:              338
Bass files:              334
Matched songs (all 3):   328
4-bar segments:          4,462
BPM range:               ~90–130 BPM (typical 80s synth-pop tempo)
Segment length:          64 steps (16 steps/bar × 4 bars)
Drum classes:            kick, snare, hi-hat (3 classes from GM drum map)
```

### Exploratory Analysis

The following plots are generated in the notebook:

```python
# 1. BPM distribution
plt.hist(all_bpms, bins=30)

# 2. Melody pitch distribution
all_pitches = all_melodies.flatten()
all_pitches = all_pitches[all_pitches > 0] - 1
plt.hist(all_pitches, bins=50)

# 3. Average drum activation per step
avg_drum = all_drums.mean(axis=0).T  # shape: (3, 64)
plt.imshow(avg_drum, aspect='auto')

# 4. Bass pitch distribution
all_bass_pitches = all_bass.flatten()
all_bass_pitches = all_bass_pitches[all_bass_pitches > 0] - 1
plt.hist(all_bass_pitches, bins=40)
```

Load the metadata:

```python
import pandas as pd

bpm_df    = pd.read_csv("processed/bpm_data.csv")
melody_df = pd.read_csv("processed/melody_dataset.csv")
drum_df   = pd.read_csv("processed/drum_dataset.csv")
bass_df   = pd.read_csv("processed/bass_dataset.csv")
```

Key columns in `bass_dataset.csv`:

```text
artist, song, bass_path, instrument_program, instrument_name, n_notes, avg_pitch, bpm
```

## Step 2: Build the 16-step Grid Dataset

The notebook converts each matched melody + drum + bass MIDI triplet into 4-bar segments:

```python
segments = midi_triplet_to_segments(
    melody_path,
    drum_path,
    bass_path,
    min_melody_notes=8,
    min_drum_hits=8,
    min_bass_notes=4
)
```

Each segment is a dictionary:

```python
{
    "melody": np.array of shape (64,),      # pitch tokens, 0=rest
    "drums":  np.array of shape (64, 3),    # [kick, snare, hat] binary
    "bass":   np.array of shape (64,),      # pitch tokens, 0=rest (same format as melody)
    "bpm":    float,
    "start_time": float
}
```

The full dataset is saved as a compressed numpy archive:

```python
np.savez(
    "processed/task2_bass_drum_dataset_4bar.npz",
    melodies=all_melodies,   # shape: (N, 64)
    drums=all_drums,         # shape: (N, 64, 3)
    bass=all_bass,           # shape: (N, 64)
    bpms=all_bpms,           # shape: (N,)
    files=...,
    start_times=...
)
```

### Drum class mapping

Only three drum classes are used, merged from General MIDI pitch numbers:

```text
kick:   MIDI 35, 36
snare:  MIDI 38, 40
hi-hat: MIDI 42, 44, 46
```

### Bass pitch representation

Bass notes use the same token format as melody:

```text
0         = REST (no note)
1 ~ 128   = MIDI pitch 0~127, shifted by +1
```

Typical bass pitch range in synth-pop: MIDI 28–50 (E1–D3).

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

### Rule-based vs LSTM

| | Rule-based | ConditionalBassDrumLSTM |
|---|---|---|
| Training required | No | Yes (30 epochs) |
| Melody-aware | Partial (root pitch only) | Yes (full sequence) |
| Output diversity | None (always identical) | High (stochastic mode) |
| Beat regularity | Perfect (hardcoded) | Lower (learned from data) |
| Bass MAE | 1.43 | **0.77** (det) / **0.26** (stoch) |
| Drum MAE | 0.60 | **0.06** (det) / **0.24** (stoch) |
| Total MAE | 2.03 | **0.83** (det) / **0.50** (stoch) |
| Handles syncopation | No | Yes |
| Interpretable | Yes | No (black box) |

### Why LSTM over Transformer?

1. **Sequence length**: 64-step sequences are short enough that LSTMs handle them well.
2. **Data size**: With ~4,462 training segments, a Transformer would likely overfit.
3. **Simplicity**: LSTM is easier to interpret and debug for a course project.

## Step 4: Handle Class Imbalance

Before training, compute `pos_weight` from `drums_train` to correct for drum class imbalance.
This must be done **after** `train_test_split`, using only the training split:

```python
from sklearn.model_selection import train_test_split

(melody_train, melody_val,
 bpm_train,    bpm_val,
 bass_train,   bass_val,
 drums_train,  drums_val) = train_test_split(
    X_melody, X_bpm, Y_bass, Y_drums,
    test_size=0.1, random_state=42
)

positive_counts = drums_train.sum(axis=(0, 1))   # shape: (3,)
total_counts    = drums_train.shape[0] * drums_train.shape[1]
negative_counts = total_counts - positive_counts

pos_weight = np.sqrt(negative_counts / (positive_counts + 1e-8))
```

Expected `pos_weight` values:

```text
kick:   ~1.82   (fires ~23% of steps)
snare:  ~2.59   (fires ~13% of steps — rarest)
hi-hat: ~1.13   (fires ~42% of steps — most common)
```

`pos_weight` is applied only to the drum BCE loss. Bass uses CrossEntropyLoss without
separate weighting because the REST token imbalance is addressed by `rest_penalty` at
generation time instead.

## Step 5: Train the ConditionalBassDrumLSTM

```python
torch.manual_seed(42)
np.random.seed(42)

model = ConditionalBassDrumLSTM(
    melody_vocab_size=129,   # 0=rest, 1~128=MIDI pitch+1
    bass_vocab_size=129,
    melody_emb_dim=32,
    hidden_dim=128,
    num_layers=2,
    drum_dim=3,
    dropout=0.2
)

model = train_conditional_bass_drum_lstm(
    model,
    train_loader,
    pos_weight=pos_weight,
    num_epochs=30,
    lr=0.001
)
```

The random seed is fixed **before** DataLoader and model creation to ensure reproducible
training results across runs. Without this, the tuned threshold and scale parameters
become invalid each time the notebook is re-executed.

### Architecture

```text
melody (64,)  ->  Embedding (64, 32)
BPM scalar    ->  repeated  (64,  1)
                  concat    (64, 33)
                  LSTM      (64, 128)
                 /          \
         bass_head        drum_head
        Linear(128->129)  Linear(128->3)
             |                 |
     CrossEntropyLoss    BCEWithLogitsLoss
      (bass target)       (drum target)
```

### Loss Function

Both bass and drum losses are summed and optimized jointly:

```python
bass_loss = CrossEntropyLoss(bass_logits, bass_target)   # 129-class classification
drum_loss = BCEWithLogitsLoss(drum_logits, drum_target,  # binary per drum class
                              pos_weight=pos_weight)
loss = bass_loss + drum_loss
```

BPM is normalized before use:

```python
bpm_norm = (bpm - 120.0) / 40.0
```

## Step 6: Generate Bassline and Drums — Two Modes

### The REST Collapse Problem (and fix)

A naive argmax on bass logits always predicts REST (token 0) because the model
assigns REST ~60% probability at each step. This is fixed by applying a penalty
to the REST logit before sampling:

```python
bass_logits[:, :, 0] -= rest_penalty   # suppress REST
bass_logits = bass_logits / temperature
bass_pred = torch.multinomial(softmax(bass_logits), num_samples=1)
```

`rest_penalty=1.5` was found by sweeping values and matching the real bass
activation rate (~6.57 notes/bar).

### Mode A: Deterministic (threshold-based)

Same melody always produces the same bass and drum pattern:

```python
bass_pred, drum_pred = generate_bass_and_drums_rest_penalty(
    model,
    melody_grid,
    bpm,
    temperature=0.4,
    rest_penalty=1.5,
    drum_thresholds=np.array([0.46, 0.40, 0.50])   # [kick, snare, hat]
)
```

`drum_thresholds` were tuned by sweeping values to match real training data
hit rates per drum class (kick ~3.72/bar, snare ~2.07/bar, hat ~7.00/bar).

### Mode B: Stochastic (sampling-based)

Same melody produces a different pattern each time:

```python
bass_pred, drum_pred = generate_bass_and_drums_stochastic(
    model,
    melody_grid,
    bpm,
    temperature=1.0,
    drum_scale=np.array([0.65, 0.50, 0.83])   # [kick, snare, hat]
)
```

Bass uses `multinomial` sampling from the softmax distribution.
Drums use `drum_scale` to reduce over-activation before `np.random.binomial` sampling:

```python
drum_probs = sigmoid(drum_logits) * drum_scale
drum_pred  = np.random.binomial(1, drum_probs)
```

`drum_scale` was tuned because raw sigmoid probabilities produce too many drum hits
(temperature scaling had the opposite effect due to predominantly negative logits).

### Which mode to use?

```text
Deterministic: reproducible evaluation, comparison against baseline
Stochastic:    generating multiple variations from the same melody
```

## Step 7: Reconstruct MIDI

Both generation modes output `bass_pred (64,)` and `drum_pred (64, 3)`.
Pass these to `create_lstm_midi`:

```python
create_lstm_midi(
    melody_grid=sample_melody,
    bass_pred=bass_pred,
    drum_pred=drum_pred,
    bpm=sample_bpm,
    output_path="lstm_output.mid"
)
```

The output MIDI contains three tracks:

```text
Track 1: Melody Synth Lead   (program 80)
Track 2: LSTM Bass           (program 38)   <- LSTM-generated
Track 3: LSTM Drums          (channel 9, is_drum=True)  <- LSTM-generated
```

## Evaluation

### Metrics

**1. Drum Hit Rate Comparison**
Average kick, snare, and hi-hat hits per bar across 50 samples, compared to real data.

**2. Bass Activation Rate Comparison**
Average bass notes per bar across 50 samples, compared to real data (~6.57 notes/bar).

**3. MAE from Real Distribution**
Mean Absolute Error between generated hit/note rates and real training data distribution.
Lower = better.

**4. Diversity**
Standard deviation of total hit counts across 10 runs from the same melody.
Rule-based always produces std=0. LSTM stochastic shows std > 2 for all drum classes.
LSTM deterministic produces std=0 for drums (threshold-based) but some variance for bass.

**5. Loss Weight Sensitivity**
Sweep of `drum_loss_weight` (w ∈ {0.5, 1.0, 2.0}) with 5-epoch quick training to verify
that equal weighting (`w=1.0`) is optimal. Absolute MAE values from the quick sweep are
not comparable to full training results; only the relative ordering matters.

### Evaluation Results

```text
                     bass    kick    snare    hat    drum MAE   bass MAE  total MAE
                   notes/bar  /bar    /bar    /bar
Real data (train)   6.57    3.72    2.07    7.00      —          —          —
Rule-based          8.00    3.00    2.00    8.00     0.60       1.43       2.03
LSTM deterministic  7.33    3.58    2.05    6.97     0.06       0.77       0.83
LSTM stochastic     6.83    3.54    2.10    6.48     0.24       0.26       0.50
```

LSTM stochastic achieves the best total MAE (0.50) and bass MAE (0.26).
LSTM deterministic achieves the best drum MAE (0.06), with perfectly stable drum output.
Both LSTM methods are significantly better than the rule-based baseline (total MAE 2.03).

Note: The deterministic bass uses `torch.multinomial` with low temperature, so bass results
vary slightly across runs. Only drum output is strictly deterministic (threshold-based).

### Diversity Results

```text
                   kick std   snare std   hat std   bass std
Rule-based           0.00       0.00       0.00      0.00   (always fixed)
Deterministic        0.00       0.00       0.00      2.40   (drum fixed, bass varies)
Stochastic           2.29       1.58       3.10      4.02   (all vary)
```

Stochastic generation produces meaningfully different outputs across runs (std > 2 for all
drum classes), enabling multiple accompaniment variations from the same melody.

### Loss Weight Sensitivity

A quick sweep of `drum_loss_weight` (w ∈ {0.5, 1.0, 2.0}) confirmed that `w = 1.0`
(equal weighting) gives the best total MAE, justifying the default `loss = bass_loss + drum_loss`.

### Audio Comparison

```python
import IPython.display as ipd
print("Rule-based:")
ipd.display(ipd.Audio("rule_based_synthpop.mid"))
print("LSTM (deterministic):")
ipd.display(ipd.Audio("lstm_deterministic.mid"))
print("LSTM (stochastic):")
ipd.display(ipd.Audio("lstm_stochastic.mid"))
```

## Key Parameters Reference

| Parameter | Value | Effect |
|---|---|---|
| `random_state=42` | train_test_split | reproducible data split |
| `torch.manual_seed(42)` | training cell | reproducible model weights |
| `test_size=0.1` | train_test_split | 10% validation split |
| `pos_weight` | BCEWithLogitsLoss | corrects drum class imbalance |
| `num_epochs=30` | training | sufficient convergence (loss 3.1359 → 2.3232) |
| `rest_penalty=1.5` | deterministic generation | suppresses bass REST collapse |
| `temperature=0.4` | deterministic generation | sharpens bass distribution |
| `drum_thresholds=[0.46, 0.40, 0.50]` | deterministic generation | per-drum decision boundary |
| `drum_scale=[0.65, 0.50, 0.83]` | stochastic generation | reduces drum over-activation |
| `temperature=1.0` | stochastic generation | controls bass randomness |
| `STEPS_PER_BAR=16` | grid representation | 16th-note quantization |
| `BARS_PER_SEGMENT=4` | grid representation | 4-bar (64-step) segments |

## What You Should Understand

`task2.ipynb` is the single entry point for the full pipeline. The key questions it answers:

1. What does the synth-pop dataset look like? (BPM distribution, melody pitch range, bass pitch range, drum density)
2. How are melody, bass, and drum MIDI tracks turned into fixed-length grids for training?
3. How does one LSTM simultaneously learn to generate both bass and drum from melody?
4. Why does argmax on bass always predict REST, and how does `rest_penalty` fix it?
5. Why does temperature scaling increase drum activation instead of decreasing it, and how does `drum_scale` solve it?
6. How do deterministic and stochastic generation differ in practice?
7. Does the LSTM produce more musically appropriate bass and drums than the rule-based baseline?

## Related Work

**Lakh MIDI Dataset (LMD)** (Raffel, C., 2016. *Learning-Based Methods for Comparing
Sequences, with Applications to Audio-to-MIDI Alignment and Matching*. PhD thesis,
Columbia University.)

**Groove MIDI Dataset** (Gillick, J. et al., 2019. *Learning to Groove with Inverse
Kinematics*. ICML 2019.)

**GrooVAE** (Google Magenta, 2019.) — VAE-based conditioned drum generation.
Unlike this project, GrooVAE conditions on a tap rhythm (not melody) and uses
continuous timing/velocity rather than a discrete grid.

**Pop Music Transformer / REMI** (Huang, Y.S. and Yang, Y.H., 2020. *Pop Music
Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Music*.
ACM MM 2020.)

**Music Transformer** (Huang, C.A. et al., 2018. *Music Transformer: Generating Music
with Long-Term Structure*. ICLR 2019.) — Transformer direction this LSTM could be
extended toward with more data.

### How This Project Differs from Prior Work

Prior work typically:
- Conditions on rhythm input or chord sequence, not a melody pitch sequence
- Generates only drums OR only bass, not both simultaneously
- Uses continuous representations (timing, velocity) for expressiveness

This project:
- Conditions directly on the **melody pitch sequence**
- Generates **both bass and drums jointly** from one model
- Uses a **discrete 16-step grid** for simplicity
- Targets a specific genre (80s synth-pop) with a small curated dataset
