"""Deterministic unit and integration checks for the PTBCC reconstruction."""

import unittest

import numpy as np

from repro.src.run_ptbcc import (
    accuracy,
    audit_claims,
    fit_ptbcc,
    load_public_datasets,
    synthetic_trial,
    vote_distribution,
)


class PTBCCReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.datasets = load_public_datasets()

    def test_public_dataset_statistics_match_paper(self) -> None:
        self.assertEqual([d.name for d in self.datasets], ["CF", "MS", "Dog", "Face", "Adult", "Web"])
        for dataset in self.datasets:
            dataset.validate()

    def test_variational_distributions_are_normalized_and_finite(self) -> None:
        fit = fit_ptbcc(self.datasets[0])
        self.assertTrue(np.isfinite(fit.phi).all())
        self.assertTrue(np.isfinite(fit.prototypes).all())
        self.assertTrue(np.isfinite(fit.worker_weights).all())
        np.testing.assert_allclose(fit.phi.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(fit.prototypes.sum(axis=2), 1.0, atol=1e-12)
        np.testing.assert_allclose(fit.worker_weights.sum(axis=1), 1.0, atol=1e-12)
        self.assertLess(fit.max_delta, 1e-3)

    def test_exact_model_synthetic_recovery_improves_truth_accuracy(self) -> None:
        row = synthetic_trial(73001)
        self.assertGreater(row["ptbcc_accuracy"], row["mv_accuracy"])
        self.assertLess(row["prototype_mae_after_matching"], 0.08)

    def test_catalog_number_claims_are_not_in_paper_source(self) -> None:
        probes = audit_claims()["text_probes"]
        self.assertTrue(probes["paper_method_name_ptbcc"])
        self.assertTrue(probes["paper_says_11_datasets"])
        self.assertTrue(probes["paper_says_15_percent"])
        self.assertFalse(probes["paper_contains_26_percent"])
        self.assertFalse(probes["paper_contains_68_73"])
        self.assertFalse(probes["paper_contains_74_11"])

    def test_public_subset_has_nontrivial_ground_truth(self) -> None:
        for dataset in self.datasets:
            self.assertGreater(len(dataset.truths), 250)
            score = accuracy(dataset, vote_distribution(dataset))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

