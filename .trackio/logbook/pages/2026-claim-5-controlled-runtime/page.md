# 2026 Claim 5 - controlled runtime


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2026_claim5_runtime", "created_at": "2026-07-23T15:55:33+00:00", "title": "Claim 5 — BLOCKED", "pinned": true, "pinned_at": "2026-07-23T15:55:33+00:00"}
-->
# Claim 5 — BLOCKED

> PTBCC uses less than 10% of the computational cost of
> confusion-matrix-based baselines while matching or exceeding their accuracy.

Five clean Python processes per method run all eleven datasets. Method order
rotates; data loading occurs before the timer; process and wall clocks are
retained; predictions must remain invariant; intervals use `10,000`
deterministic bootstrap draws.

| Ratio | Process-time median | 95% bootstrap interval |
|---|---:|---:|
| PTBCC / FGBCC | `0.2258383` | `[0.2007075, 0.2294063]` |
| PTBCC / released DS | `4.3310585` | recorded in raw evidence |
| PTBCC / BWA | `7.4004689` | recorded in raw evidence |

PTBCC exceeds FGBCC accuracy by `2.9570868` percentage points, aligning with
the accuracy side of the statement. Every controlled runtime sample exceeds
the paper's `0.10` threshold on this authorized local Apple CPU.

This is a rigorous setup-specific contradiction, but not a paper-level
falsification: the paper used a 2-vCPU Intel Xeon Platinum 8369HC, reports no
raw timing samples or exact aggregate denominator, and its plot reflects
original baseline loops rather than the equation-equivalent vectorized FGBCC
used here.

Independent negative control: injecting a zero-time sample is rejected.

Machine evidence:
`.openresearch/artifacts/claim_5/raw_runtime_benchmark.json`,
`claim_contract.json`, `independent_checker.json`,
`negative_control.json`, and `EVAL.md`.
