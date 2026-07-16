# Methods


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_747409a4d01e", "created_at": "2026-07-16T15:10:41+00:00", "title": "Protocol"}
-->
# Methods

- Paper: OpenReview `KJq0iScNM6`, arXiv `2508.02123v1`; exact source hash recorded.
- Data: CF/MS at pinned active-crowd-toolkit commit; Dog/Face/Adult from the cited public archive (archive hash recorded); Web at pinned SpectralMethodsMeetEM commit. Dataset dimensions exactly match Table 3.
- Model: mean-field coordinate updates from equations 7–14; majority vote initialization; published accurate/adversarial prototypes.
- Controls: Dawid–Skene baseline, 40 generated-model recovery seeds, prototype matching, normalization/finite checks, literal source probes.
- Scope: six of eleven exact datasets were publicly recoverable; five are explicitly excluded. CPU only, 76.1 seconds.
