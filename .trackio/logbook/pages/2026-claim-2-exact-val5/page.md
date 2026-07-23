# 2026 Claim 2 - exact Val5


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2026_claim2_val5", "created_at": "2026-07-23T15:55:33+00:00", "title": "Claim 2 — VERIFIED", "pinned": true, "pinned_at": "2026-07-23T15:55:33+00:00"}
-->
# Claim 2 — VERIFIED

> PTBCC achieves up to 15 percentage points of accuracy improvement over the
> best baseline on Val5 (Table 4).

## Claim contract

Use the exact Val5 corpus dimensions from Table 3 (`100` tasks, `38`
annotators, `5` classes, `1,000` labels, `100` truths), fractional-tie
accuracy, and faithful released or paper-referenced baselines. The verifier
requires PTBCC accuracy `0.56`, the strongest baseline `0.41`, and an exact
gain of `0.15`.

## Regenerated evidence

| Method | Accuracy |
|---|---:|
| MV | `0.3516666667` |
| BWA | `0.35` |
| FGBCC | `0.38` |
| released DS | `0.41` |
| PTBCC | **`0.56`** |

The direct difference is `0.56 - 0.41 = 0.15`, reproducing the paper's
headline result on the exact dataset.

Independent negative control: changing PTBCC from `0.56` to `0.55` is rejected.

Machine evidence:
`.openresearch/artifacts/claim_2/raw_val5_result.json`,
`claim_contract.json`, `independent_checker.json`,
`negative_control.json`, and `EVAL.md`.
