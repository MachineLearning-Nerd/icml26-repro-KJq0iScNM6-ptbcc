# 2026 Claim 4 - prototype-count ablation


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2026_claim4_ablation", "created_at": "2026-07-23T15:55:33+00:00", "title": "Claim 4 — BLOCKED", "pinned": true, "pinned_at": "2026-07-23T15:55:33+00:00"}
-->
# Claim 4 — BLOCKED

> Table 5 reports a peak at `|S|=2` (`0.7472`) followed by `0.7300` at
> `|S|=3` and `0.7271` at `|S|=4`.

Every configuration is rerun on all eleven datasets. `S=3` and `S=4` use
three committed Dirichlet seeds (`20260715–20260717`); `3-Ran` is a separate
uniform-matrix control.

| Configuration | Regenerated macro accuracy |
|---|---|
| `S=2` | `0.7471525427` |
| `S=3`, three seeds | `0.7420771316`, `0.7304916653`, `0.7283875737` |
| `S=4`, three seeds | `0.7362274154`, `0.7138656936`, `0.7179507310` |
| `3-Ran`, uniform | `0.7287995247` |

`S=2` is the peak in every tested seed, directly supporting the qualitative
ordering. No seed reproduces the paper's exact `S=3/S=4` pair, and the paper
does not report its seed. The exact-number claim therefore remains BLOCKED.

Independent negative control: relabeling `3-Ran` as a Dirichlet initialization
is rejected.

Machine evidence:
`.openresearch/artifacts/claim_4/raw_ablation.json`,
`claim_contract.json`, `independent_checker.json`,
`negative_control.json`, and `EVAL.md`.
