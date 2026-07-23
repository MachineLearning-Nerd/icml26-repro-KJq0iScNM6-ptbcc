# 2026 claim-by-claim campaign


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2026_campaign_summary", "created_at": "2026-07-23T15:55:33+00:00", "title": "Claim-by-claim release summary", "pinned": true, "pinned_at": "2026-07-23T15:55:33+00:00"}
-->
# Exact-corpus cumulative result

This additive campaign preserves the complete judged revision
`e57f7e6e348fea6c5a0467ca33f94375b5bf2623` and extends it from six to all
eleven paper datasets. It adds BWA, the released Dawid–Skene implementation,
golden-output-validated FGBCC, seeded prototype-count ablations, controlled
process timing, claim contracts, and an independent file-only checker.

| Claim | Result | Direct evidence |
|---|---|---|
| Shared prototype architecture | **VERIFIED** | Normalized `[S,K,K]` prototypes and `[W,S]` annotator mixtures; 40/40 synthetic wins; mean matched prototype MAE `0.0417523` |
| Up to 15-point gain on Val5 | **VERIFIED** | PTBCC `0.56`, released DS `0.41`; exact gain `0.15` |
| Exact four-number Table 4 conjunction | **BLOCKED** | MV, BWA, and PTBCC match at four decimals; FGBCC regenerates `0.7176`, not printed `0.7175`; paper-side numerical provenance is absent |
| Exact prototype-count ablation | **BLOCKED** | `S=2` peaks in all three committed seeds; exact `S=3/S=4` pair is seed-sensitive and the paper seed is absent |
| Under 10% computational cost | **BLOCKED** | Local PTBCC/FGBCC process-time ratio `0.2258 [0.2007, 0.2294]`; paper gives no raw times or precise aggregate denominator and used a different CPU/implementation stack |

`BLOCKED` is not a proxy pass. It records that the available direct experiment
is complete but the paper omits information required to verify or falsify the
exact statement.

Formal command on every node:

```bash
uv sync --frozen && uv run python repro/src/run_ptbcc.py --output-dir outputs/full && uv run python -m unittest -v repro.tests.test_ptbcc
```

Compute: local CPU only, no GPU and no Hugging Face cpu-upgrade. Accepted
cumulative verifier commit:
`dc8d5eba7a9cf92fe64a0e75473df740146a191f`.

Detailed illustrated report:
https://github.com/MachineLearning-Nerd/icml26-repro-KJq0iScNM6-ptbcc/blob/orx/release-candidate-report-and-additive-logbook/reports/ptbcc-claim-by-claim-2026-07-23/report.md
