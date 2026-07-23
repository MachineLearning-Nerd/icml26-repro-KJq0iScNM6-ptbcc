# 2026 Claim 3 - exact Table 4


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2026_claim3_table4", "created_at": "2026-07-23T15:55:33+00:00", "title": "Claim 3 — BLOCKED", "pinned": true, "pinned_at": "2026-07-23T15:55:33+00:00"}
-->
# Claim 3 — BLOCKED

> Across eleven datasets, Table 4 reports PTBCC `0.7472`, FGBCC `0.7175`,
> BWA `0.7132`, and MV `0.6986`.

All eleven corpora match the paper's Table 3 counts. The calculation is the
unweighted macro mean of per-dataset fractional-tie accuracies.

| Method | Paper | Regenerated | Four-decimal result |
|---|---:|---:|---|
| MV | `0.6986` | `0.6986047829` | `0.6986` |
| BWA | `0.7132` | `0.7131502844` | `0.7132` |
| FGBCC | `0.7175` | `0.7175816744` | `0.7176` |
| PTBCC | `0.7472` | `0.7471525427` | `0.7472` |

The FGBCC implementation reproduces the authors' exact Aircr golden output,
but the full macro differs from the printed value by `0.0000816744`. The
paper's complete numerical provenance and PTBCC author code are unavailable,
so this four-number conjunction cannot honestly be promoted to VERIFIED or
FALSIFIED.

Independent negative control: removing any one dataset is rejected.

Machine evidence:
`.openresearch/artifacts/claim_3/raw_table4_results.json`,
`claim_contract.json`, `independent_checker.json`,
`negative_control.json`, and `EVAL.md`.
