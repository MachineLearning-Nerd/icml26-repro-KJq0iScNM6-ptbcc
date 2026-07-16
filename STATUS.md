# STATUS — PTBCC (KJq0iScNM6)

**Session:** autoloop · **State:** publish gate · **Updated:** 2026-07-16

## Claims

1. “CPBCC yields up to 26% accuracy improvement” — **falsified:** source says PTBCC and 15%; no 26%.
2. “Boosts average accuracy from 68.73% to 74.11% across 10 datasets” — **falsified:** source says 69.86% → 74.72% across 11.
3. “Models annotators through class-specific prototypes and annotator-specific weights” — **verified:** explicit reconstructed tensors/weights, 40/40 synthetic wins, six exact public datasets.

## Current step

Full CPU run completed in 76.1 seconds. Five tests pass. PTBCC macro accuracy
is 79.31% vs majority vote 76.57% and Dawid–Skene 77.33%; generated-model
recovery wins 40/40 trials with prototype MAE 0.04175.

## Next

Secret-scan, publish the HF Space and public GitHub repository, then await the
official verdict.
