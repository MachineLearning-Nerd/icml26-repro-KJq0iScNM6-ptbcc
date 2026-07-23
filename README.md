# PTBCC claim-by-claim reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-KJq0iScNM6-ptbcc/blob/main/notebooks/ptbcc_claim_by_claim.py)

This is a CPU-only reproduction of *Let the Prototype Guide You: Robust
Aggregation of Sparse Multi-Class Annotations via Annotator Prototype
Learning* (arXiv `2508.02123`). It tests the paper's shared-prototype
architecture, Val5 headline gain, 11-dataset Table 4 averages,
prototype-count ablation, and runtime claim.

The strongest result is directly reproduced: PTBCC reaches `0.56` on the exact
Val5 corpus versus `0.41` for the strongest faithfully released baseline,
matching the paper's `+15.0` percentage-point gain. The final evidence verdicts
are:

- **VERIFIED:** shared-prototype architecture and Val5 gain.
- **BLOCKED:** the exact Table 4 conjunction, exact stochastic ablation
  values, and paper-level runtime statement. Each is narrowed to a documented
  missing seed, numerical provenance, or timing denominator—not replaced by a
  proxy result.

The campaign uses all 11 exact datasets, author-released or paper-referenced
baselines, deterministic seeds, five clean-process timing repetitions,
independent file-only verification, and one deliberate corruption per claim.
All runs used local CPU; Hugging Face cpu-upgrade and GPU hardware were not
used. Read the [illustrated technical report](reports/ptbcc-claim-by-claim-2026-07-23/report.md)
or the [self-contained marimo tutorial](notebooks/ptbcc_claim_by_claim.py).

## Experiment log

The command shown below is copied verbatim from each formal node's
`orx exp status`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`main`](https://github.com/MachineLearning-Nerd/icml26-repro-KJq0iScNM6-ptbcc/tree/main) | Validated starting point; reserved as the publication surface | Not run as an experiment (publication surface) | Starting judge evidence preserved | — |
| [`orx/runnable-frozen-ptbcc-baseline`](https://github.com/MachineLearning-Nerd/icml26-repro-KJq0iScNM6-ptbcc/tree/orx/runnable-frozen-ptbcc-baseline) | Locked `uv` environment and runnable six-dataset baseline | `uv sync --frozen && uv run python repro/src/run_ptbcc.py --output-dir outputs/full && uv run python -m unittest -v repro.tests.test_ptbcc` | Baseline passed; Claim 1 mechanism retained | Local CPU |
| [`orx/exact-ds-full-fgbcc-and-logged-ablation`](https://github.com/MachineLearning-Nerd/icml26-repro-KJq0iScNM6-ptbcc/tree/orx/exact-ds-full-fgbcc-and-logged-ablation) | Exact 11-dataset corpus, released DS/BWA, full FGBCC, seeded ablation | `uv sync --frozen && uv run python repro/src/run_ptbcc.py --output-dir outputs/full && uv run python -m unittest -v repro.tests.test_ptbcc` | Val5 verified; three Table 4 values match at four decimals; exact ablation values remain seed-sensitive | Local CPU |
| [`orx/process-isolated-cpu-runtime-benchmark`](https://github.com/MachineLearning-Nerd/icml26-repro-KJq0iScNM6-ptbcc/tree/orx/process-isolated-cpu-runtime-benchmark) | Five clean-process repeats, rotated order, process/wall clocks, bootstrap intervals | `uv sync --frozen && uv run python repro/src/run_ptbcc.py --output-dir outputs/full && uv run python -m unittest -v repro.tests.test_ptbcc` | PTBCC/FGBCC CPU ratio `0.2258 [0.2007, 0.2294]`; exact paper claim blocked by unspecified denominator and different stack | Local CPU |
| [`orx/cumulative-claim-verifier-and-evidence-bundle`](https://github.com/MachineLearning-Nerd/icml26-repro-KJq0iScNM6-ptbcc/tree/orx/cumulative-claim-verifier-and-evidence-bundle) | Claim contracts, independent checker, negative controls, cumulative regression | `uv sync --frozen && uv run python repro/src/run_ptbcc.py --output-dir outputs/full && uv run python -m unittest -v repro.tests.test_ptbcc` | Claims 1–2 VERIFIED; Claims 3–5 BLOCKED; all five controls rejected; 9/9 tests passed | Local CPU |

## Reproduce

The one fixed command used by every experiment node is:

```bash
uv sync --frozen && uv run python repro/src/run_ptbcc.py --output-dir outputs/full && uv run python -m unittest -v repro.tests.test_ptbcc
```

It recreates the machine-readable outputs, independently verifies every claim,
runs all negative controls, rebuilds the five report figures, and executes the
cumulative test suite using the repository-level `.venv` and locked Python
environment.
