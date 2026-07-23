# 2026 cumulative verifier


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2026_cumulative_verifier", "created_at": "2026-07-23T15:55:33+00:00", "title": "Independent checker and controls", "pinned": true, "pinned_at": "2026-07-23T15:55:33+00:00"}
-->
# Independent checker and controls

The checker runs in a separate Python process and reads only emitted CSV/JSON
evidence. It recomputes every statistic and exits nonzero if a claim contract
is incomplete, if accepted evidence changes, or if a deliberate corruption
survives.

| Claim | Deliberate corruption | Result |
|---|---|---|
| 1 | Remove one required synthetic seed | rejected |
| 2 | Change Val5 PTBCC accuracy from `0.56` to `0.55` | rejected |
| 3 | Remove one of eleven datasets | rejected |
| 4 | Mislabel the `3-Ran` uniform control | rejected |
| 5 | Inject a zero wall-time sample | rejected |

Final machine verdict set:

```json
{
  "claim_1": "VERIFIED",
  "claim_2": "VERIFIED",
  "claim_3": "BLOCKED",
  "claim_4": "BLOCKED",
  "claim_5": "BLOCKED"
}
```

All five evidence-validity checks pass, all five controls are rejected, and
the cumulative verifier run passes all nine repository tests. Source retrieval,
hashes, section anchors, assumptions, fixed command, environment, deterministic
seeds, CPU provenance, limitations, and deviations are retained under
`.openresearch/artifacts/` and `repro/audit/`.

No score increase is claimed until a live judge evaluates a published
revision.
