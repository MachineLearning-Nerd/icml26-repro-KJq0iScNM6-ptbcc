# BLOCKED

Five clean-process, rotated-order repetitions on all 11 datasets reproduce the
paper's approximately three-point accuracy gain (`2.9571` percentage points)
but contradict the `<10%` timing threshold on this local implementation stack:

| Comparator | PTBCC/comparator process-time median | Bootstrap 95% interval |
|---|---:|---:|
| FGBCC | 0.2258 | [0.2007, 0.2294] |
| released DS | 4.3311 | [3.5987, 4.9387] |
| BWA | 7.4005 | [6.2888, 7.5679] |

The corresponding PTBCC/FGBCC wall-time median is `0.2239`, interval
`[0.1778, 0.2840]`. Every individual ratio sample exceeds `0.10`, predictions
are invariant across worker processes, and the injected zero-time negative
control is rejected.

The paper nevertheless omits the aggregate denominator and raw timings, used a
2-vCPU Intel Xeon Platinum 8369HC, and plotted original baseline loops. This
run uses Apple CPU hardware and an Aircr-golden-equivalent vectorized FGBCC.
It is a rigorous contradiction on the authorized local stack, but not a
faithful reproduction of the paper's timing environment; the paper-level claim
therefore remains `BLOCKED`, not `FALSIFIED`.

See `raw_runtime_benchmark.json`, `independent_checker.json`,
`negative_control.json`, and `run_provenance.json`.
