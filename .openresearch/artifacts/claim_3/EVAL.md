# BLOCKED

The exact 11-dataset macro regenerates as:

| Method | Paper | Observed | Four-decimal match |
|---|---:|---:|:---:|
| MV | 0.6986 | 0.6986047829 | yes |
| BWA | 0.7132 | 0.7131502844 | yes |
| FGBCC | 0.7175 | 0.7175816744 | no (`0.7176`) |
| PTBCC | 0.7472 | 0.7471525427 | yes |

FGBCC exactly matches the authors' released Aircr golden result, but its
11-dataset macro differs from the printed value by `0.0000816744`. PTBCC has no
released author implementation or numerical environment. The entire
four-number conjunctive claim therefore remains `BLOCKED`; three matching
numbers are not promoted to a full pass, and the tiny FGBCC discrepancy is not
promoted to a falsification without full author-side provenance.

Dropping one dataset invalidates the independent checker, as shown in
`negative_control.json`. See `raw_table4_results.json` and
`independent_checker.json`.
