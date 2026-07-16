# Negative controls


---
<!-- trackio-cell
{"type": "code", "id": "cell_d15f96a86163", "created_at": "2026-07-16T15:09:40+00:00", "title": "Run: python (exit 0)", "command": ["python", "-m", "unittest", "discover", "-v", "-s", "repro/tests"], "exit_code": 0, "duration_s": 1.681}
-->
````bash
$ python -m unittest discover -v -s repro/tests
````

exit 0 · 1.7s


````output
test_catalog_number_claims_are_not_in_paper_source (test_ptbcc.PTBCCReproductionTests.test_catalog_number_claims_are_not_in_paper_source) ... ok
test_exact_model_synthetic_recovery_improves_truth_accuracy (test_ptbcc.PTBCCReproductionTests.test_exact_model_synthetic_recovery_improves_truth_accuracy) ... ok
test_public_dataset_statistics_match_paper (test_ptbcc.PTBCCReproductionTests.test_public_dataset_statistics_match_paper) ... ok
test_public_subset_has_nontrivial_ground_truth (test_ptbcc.PTBCCReproductionTests.test_public_subset_has_nontrivial_ground_truth) ... ok
test_variational_distributions_are_normalized_and_finite (test_ptbcc.PTBCCReproductionTests.test_variational_distributions_are_normalized_and_finite) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.819s

OK

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_08362fbc0374", "created_at": "2026-07-16T15:10:42+00:00", "title": "Why false positives are unlikely"}
-->
The five-test suite checks exact dataset dimensions, normalized finite distributions, convergence, synthetic prototype recovery, and absence/presence of the audited source strings. The Dawid–Skene comparator separates generic crowd aggregation from the prototype mechanism. Majority vote is not universally beaten per dataset (CF is an honest exception), while the macro direction and 40/40 generated-model recovery support the specific structural claim.
