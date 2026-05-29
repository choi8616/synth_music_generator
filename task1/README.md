# Task 1: Symbolic Unconditioned Generation

Final Task 1 work is contained in:

```text
task1_final.ipynb
```

The notebook loads cleaned synth-pop data from `../synthpop_data/processed`, trains a Markov baseline and an LSTM event model, evaluates them, and writes the generated MIDI output.

For portability, the notebook can run from either:

1. `synthpop_data/processed/melodies_all/` and `synthpop_data/processed/drums_all/`, or
2. the precomputed symbolic token dataset `synthpop_data/processed/task1_combined_grid_dataset.npz`.

The second option keeps the GitHub repo small and avoids committing hundreds of MIDI files.

Final generated Task 1 MIDI is copied to the repository root as:

```text
../symbolic_unconditioned.mid
```
