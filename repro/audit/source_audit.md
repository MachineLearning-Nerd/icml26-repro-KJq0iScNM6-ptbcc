# Primary-source audit

- Paper identifier: arXiv `2508.02123v1`
- arXiv title: *Understanding the Essence: Delving into Annotator Prototype
  Learning for Multi-Class Annotation Aggregation*
- Challenge/OpenReview title: *Let the Prototype Guide You: Robust Aggregation
  of Sparse Multi-Class Annotations via Annotator Prototype Learning*
- Retrieved: `2026-07-23T13:39:50Z`
- URL: `https://ar5iv.labs.arxiv.org/html/2508.02123`
- Retrieval User-Agent:
  `OpenResearch-Reproduction/1.0 (paper audit; contact via repository)`
- HTML SHA-256:
  `1910a8d019a88a17bc2c17441a41850645cf29d99bec84c19bc61f50cc5d500b`
- Committed TeX SHA-256:
  `2b1b3d6f44c74ed16172b5e473fb75078be33e568b5c39b0d49befa25f8d3c4b`

## Anchors and exact scope

- Architecture: Method, generative process and equations (7)–(14);
  ar5iv sections `#Sx4` and parameter updates.
- Datasets and assumptions: Table 3 and Experimental Setup; exactly 11
  real-world datasets, accuracy as the metric, `|S|=2`, `e=1`, `f=5`,
  `m=1.35`, convergence threshold `xi=0.001`, and a 2-vCPU Intel Xeon
  Platinum 8369HC / 8 GB reference machine.
- Claim 2: Val5 case study `#Sx5.SSx3`, especially `#Sx5.SSx3.p4`.
  The source states a 15% accuracy improvement on Val5. Table 2 reports the
  strongest listed Val5 baseline as DS at 0.41, while Figure 2 is the only
  source for PTBCC's per-dataset value.
- Claim 3: Table 4 `#Sx5.T4`; arithmetic macro averages over all 11 named
  datasets: PTBCC 0.7472, FGBCC 0.7175, BWA 0.7132, MV 0.6986.
- Claim 4: Ablation Table 5 `#Sx5.T5` and interpretation
  `#Sx5.SSx4.p3`; all-11-dataset macro accuracies at `|S|=2,3,4` are
  0.7472, 0.7300, and 0.7271. Additional prototypes are initialized from a
  uniform Dirichlet distribution; `3-Ran` is a distinct all-entries-uniform
  matrix.
- Claim 5: Efficiency section `#Sx5.SSx5`, especially `#Sx5.SSx5.p3`.
  The prose says PTBCC uses less than 10% of the running time while improving
  average accuracy by 3%. Figure 5 compares all methods on all 11 datasets;
  the paper does not publish its underlying numeric runtime table or define a
  single named denominator in the final sentence.

Percentage language is treated as percentage points only where the table and
case-study arithmetic establish that interpretation; ambiguous quantifiers are
not silently strengthened.

## Exact corpus and baseline provenance

- The cleaned 11-dataset collection is from the FGBCC authors' repository,
  `JuJuCHEN-HHU/CodeForFGBCC`, commit
  `e2ca2b8a876bf9cceb871e8cec9081870a30aab4`. Every file is downloaded from a
  commit-pinned raw URL and checked against `repro/audit/public_data_manifest.json`.
  All 11 observed `(tasks, workers, truths, classes, labels)` tuples exactly
  equal Table 3.
- The FGBCC reference source is `code/FGBCC/method.py` at that commit, SHA-256
  `6e8e1545c950c4895165eb1aa3bc37e06097e6fa7887fe9d387e1a9e7b091979`.
  The repository publishes an Aircr accuracy of `0.8448566610455311`, used as
  a golden equivalence test before full-scale execution.
- BWA is specified by Appendix A of arXiv `1902.08918` and its MIT-licensed
  reference repository `yuan-li/truth-inference-at-scale`, commit
  `621789b2d57324d3559dc973b2613d2296d73f55`. The reference `code/bwa.py`
  SHA-256 is
  `96cd391294664de983e8c8af340f4dec9cfca322f35bb0562ab086cef5985151`.
- The published evaluation utility assigns fractional credit to tied maxima.
  On the exact corpus, this convention produces MV macro accuracy
  `0.6986047829`, which rounds to Table 4's `0.6986`; first-index tie breaking
  does not. All claim contracts therefore use fractional tie credit.
- The paper's DS baseline comes from the public Truth Inference in
  Crowdsourcing survey bundle
  `https://zhydhkcws.github.io/crowd_truth_inference/truth_inference_crowd.zip`,
  retrieved with an explicit User-Agent on `2026-07-23`, SHA-256
  `fcb72da704bf06701ebec5f47e3d85b583354a098e4410a29750abdaaa59d9a2`.
  Its `methods/c_EM/method.py` specifies uniform class priors, diagonal
  initialization `0.7`, and exactly 20 EM iterations. The previous adaptive
  convergence implementation is retained only as a diagnostic and is not used
  for paper-number contracts.
