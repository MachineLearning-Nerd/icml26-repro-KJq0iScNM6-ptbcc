# Claim 1 method

The reconstruction directly materializes arrays `[S,K,K]` and `[W,S]`, checks
normalization and finiteness, and performs Hungarian-matched prototype recovery
on data sampled from the same graphical model. The accepted baseline run is
rerun cumulatively on every descendant.
