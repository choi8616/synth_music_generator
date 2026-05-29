# Processed Synth-Pop Data

This folder contains the lightweight data needed by `task1/task1_final.ipynb`.

Commit these files:

```text
bpm_data.csv
drum_dataset.csv
melody_dataset.csv
song_metadata.csv
usable_songs.csv
task1_combined_grid_dataset.npz
```

The `.npz` file stores the symbolic REST/HOLD/PITCH + drum-mask event sequences extracted from the processed melody/drum MIDI files. With this file present, the Task 1 notebook can train and generate music without committing the MIDI folders.

Do not commit these folders unless your team/class repo is private and you have confirmed it is acceptable:

```text
melodies_all/
drums_all/
```
