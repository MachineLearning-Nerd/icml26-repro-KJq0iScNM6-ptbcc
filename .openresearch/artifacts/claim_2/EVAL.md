# VERIFIED

On the exact Val5 corpus (100 tasks, 38 workers, 5 classes, 1,000 labels, 100
truths), PTBCC regenerates accuracy `0.56`. The strongest faithfully released
baseline is the survey implementation of Dawid–Skene at `0.41` (FGBCC
`0.38`, BWA `0.35`, tie-aware MV `0.3516667`), giving a direct `15.0`
percentage-point gain.

Changing PTBCC accuracy to `0.55` makes the verifier withhold `VERIFIED`, as
recorded in `negative_control.json`. See `raw_val5_result.json`,
`independent_checker.json`, and `run_provenance.json`.
