# BLOCKED

The full 11-dataset ablation gives `S=2` macro accuracy `0.7471525427`, which
is the peak for every tested stochastic seed. The regenerated Dirichlet(1)
macros are:

| Seed | S=3 | S=4 |
|---:|---:|---:|
| 20260715 | 0.7420771316 | 0.7362274154 |
| 20260716 | 0.7304916653 | 0.7138656936 |
| 20260717 | 0.7283875737 | 0.7179507310 |

The distinct `3-Ran` uniform-matrix control is `0.7287995247`. No tested
single seed reproduces both paper values `0.7300` and `0.7271`, and the paper
does not disclose its seed. Thus the qualitative peak is robustly supported,
but the exact-number claim remains `BLOCKED`.

Mislabeling the uniform control as a Dirichlet draw invalidates the checker.
See `raw_ablation.json`, `independent_checker.json`, and
`negative_control.json`.
