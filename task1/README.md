# Task 1: Symbolic Unconditioned Music Generation

This folder is the foundation for your Task 1 pipeline.

The project goal is:

```text
clean synth-pop MIDI files
  -> REMI token sequences
  -> Markov baseline
  -> LSTM next-token model
  -> generated MIDI
```

Right now, this foundation includes the data/token pipeline and the Markov baseline. The LSTM will use the same processed token files next.

## Big Picture

The central idea is next-token prediction:

```text
Given previous music tokens, predict the next music token.
```

For example, a MIDI file becomes a sequence like:

```text
Bar_None, Position_0, Pitch_60, Velocity_127, Duration_0.4.8, ...
```

The Markov baseline learns this with count tables:

```text
p(next_token | previous 3 tokens)
```

The LSTM will learn the same task with neural network weights:

```text
p(next_token | all previous tokens summarized in hidden state)
```

## Folder Layout

```text
task1_symbolic_unconditioned/
  data/raw/                 # later: teammate's cleaned Lakh synth-pop MIDI files
  processed/                # tokenized dataset files
  outputs/                  # trained baselines and generated MIDI
  scripts/
    prepare_tokens.py       # MIDI folder -> token files
    train_markov.py         # train n-gram baseline
    generate_markov.py      # sample a MIDI from baseline
    train_lstm.py           # train the main LSTM next-token model
    generate_lstm.py        # sample a MIDI from the LSTM
    smoke_test.py           # check processed files exist
  src/task1_musicgen/
    data_pipeline.py        # reusable MIDI/token functions
    markov.py               # reusable Markov model class
    lstm_model.py           # reusable LSTM model and sampling helpers
```

## Step 1: Tokenize MIDI Data

For now, test with the Homework 3 `PDMX_subset`. Later replace the data path with the cleaned Lakh synth-pop folder.

From this folder:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python3 scripts/prepare_tokens.py \
  --data-dir /Users/choi/Desktop/SP26/CSE153/homework3/PDMX_subset \
  --processed-dir processed \
  --vocab-size 1000
```

On this machine, `/opt/anaconda3/bin/python3` is the Python environment that already has `miditok`, `symusic`, and `torch`. If your terminal environment changes later, any Python with those packages installed is fine.

Expected outputs:

```text
processed/tokenizer.json
processed/train_sequences.npy
processed/val_sequences.npy
processed/train_tokens.npy
processed/val_tokens.npy
processed/vocab_info.json
processed/tokenization_report.json
```

The important file for your presentation is `tokenization_report.json`: it records how many MIDI files were loaded, how many failed, sequence lengths, and vocabulary size.

## Step 2: Train Markov Baseline

```bash
PYTHONPATH=src /opt/anaconda3/bin/python3 scripts/train_markov.py \
  --processed-dir processed \
  --model-out outputs/markov_order4.pkl \
  --order 4
```

`--order 4` means:

```text
p(next token | previous 3 tokens)
```

This is stronger than the Homework 3 pitch-only Markov chain because it models the full REMI token stream, including bars, positions, pitches, velocities, and durations.

## Step 3: Generate a MIDI From the Baseline

```bash
PYTHONPATH=src /opt/anaconda3/bin/python3 scripts/generate_markov.py \
  --processed-dir processed \
  --model-path outputs/markov_order4.pkl \
  --out-midi outputs/symbolic_unconditioned_markov.mid \
  --max-length 512 \
  --seed 42 \
  --temperature 1.0 \
  --top-k 20
```

Important generation controls:

- `--seed`: same seed gives same sample; different seed gives a different sample.
- `--temperature`: lower is safer/repetitive; higher is more chaotic.
- `--top-k`: samples only from the top k likely next tokens.

## Step 4: Quick Health Check

```bash
PYTHONPATH=src /opt/anaconda3/bin/python3 scripts/smoke_test.py --processed-dir processed
```

## Step 5: LSTM Main Model

The LSTM uses the same `processed/train_sequences.npy` and `processed/val_sequences.npy`.

For a quick test run:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python3 scripts/train_lstm.py \
  --processed-dir processed \
  --model-out outputs/lstm_model.pt \
  --epochs 1 \
  --context-length 64 \
  --max-train-windows 128 \
  --max-val-windows 32
```

For a real run later, remove the `--max-train-windows` and `--max-val-windows` limits and increase epochs:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python3 scripts/train_lstm.py \
  --processed-dir processed_lakh \
  --model-out outputs/lstm_lakh.pt \
  --epochs 20 \
  --context-length 128 \
  --batch-size 32
```

Generate from the LSTM:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python3 scripts/generate_lstm.py \
  --processed-dir processed \
  --model-path outputs/lstm_model.pt \
  --out-midi outputs/symbolic_unconditioned_lstm.mid \
  --max-new-tokens 512 \
  --seed 42 \
  --temperature 1.0 \
  --top-k 20
```

## When Cleaned Lakh Data Is Ready

Put or point to the cleaned MIDI folder, then rerun only Step 1:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python3 scripts/prepare_tokens.py \
  --data-dir /path/to/cleaned_lakh_synthpop_midis \
  --processed-dir processed_lakh \
  --vocab-size 1000
```

Then train/generate with `--processed-dir processed_lakh`.

## What You Should Understand First

`prepare_tokens.py` is the foundation. It answers:

1. Can we load the cleaned MIDI files?
2. Can we tokenize them into symbolic music tokens?
3. How large is the vocabulary?
4. How long are the sequences?
5. Do we have train/validation data for both Markov and LSTM?

Once this works, the LSTM is just another model trained on the same token sequences.
