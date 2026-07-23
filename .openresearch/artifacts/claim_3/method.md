# Claim 3 method

Commit-pin and hash every label/truth file; validate every Table 3 dimension;
run paper-faithful MV, released DS, BWA, FGBCC, and PTBCC; record per-dataset scores before
forming an unweighted macro. The MV macro is an independent metric/data sanity
check because it must reproduce `0.6986` without learned parameters.
