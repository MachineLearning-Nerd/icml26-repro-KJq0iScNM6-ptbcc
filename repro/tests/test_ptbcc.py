"""Deterministic unit and integration checks for the PTBCC reconstruction."""

import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from repro.src.run_ptbcc import (
    accuracy,
    audit_claims,
    fit_ptbcc,
    load_paper_datasets,
    synthetic_trial,
    vote_distribution,
)
from repro.src.reference_baselines import fit_bwa, fit_published_ds


class PTBCCReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.datasets = load_paper_datasets()

    def test_public_dataset_statistics_match_paper(self) -> None:
        self.assertEqual(
            [d.name for d in self.datasets],
            [
                "Val7",
                "Aircr",
                "CF",
                "Fact",
                "MS",
                "Dog",
                "Face",
                "Adult",
                "Senti",
                "Val5",
                "Web",
            ],
        )
        for dataset in self.datasets:
            dataset.validate()

    def test_tie_aware_majority_vote_reproduces_table4(self) -> None:
        scores = [
            accuracy(dataset, vote_distribution(dataset))
            for dataset in self.datasets
        ]
        self.assertAlmostEqual(float(np.mean(scores)), 0.6986, places=4)
        val5 = next(dataset for dataset in self.datasets if dataset.name == "Val5")
        self.assertAlmostEqual(
            accuracy(val5, vote_distribution(val5)), 0.352, places=3
        )

    def test_negative_control_rejects_corrupted_table3_dimension(self) -> None:
        dataset = self.datasets[0]
        corrupted = replace(
            dataset,
            expected=(
                dataset.n_tasks + 1,
                dataset.n_workers,
                len(dataset.truths),
                dataset.n_classes,
                dataset.n_labels,
            ),
        )
        with self.assertRaisesRegex(ValueError, "expected"):
            corrupted.validate()

    def test_released_baselines_reproduce_val5_table(self) -> None:
        val5 = next(dataset for dataset in self.datasets if dataset.name == "Val5")
        self.assertAlmostEqual(accuracy(val5, fit_published_ds(val5).scores), 0.41)
        self.assertAlmostEqual(accuracy(val5, fit_bwa(val5).scores), 0.35)

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

    def test_full_corpus_has_nontrivial_ground_truth(self) -> None:
        for dataset in self.datasets:
            self.assertGreaterEqual(len(dataset.truths), 100)
            score = accuracy(dataset, vote_distribution(dataset))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_independent_claim_verifier_and_controls_passed(self) -> None:
        root = Path(__file__).resolve().parents[2]
        result_path = root / "outputs" / "full" / "claim_verdicts.json"
        self.assertTrue(
            result_path.is_file(),
            "the fixed command must run the independent checker before tests",
        )
        result = json.loads(result_path.read_text())
        self.assertEqual(
            result["verdicts"],
            {
                "claim_1": "VERIFIED",
                "claim_2": "VERIFIED",
                "claim_3": "BLOCKED",
                "claim_4": "BLOCKED",
                "claim_5": "BLOCKED",
            },
        )
        self.assertTrue(all(result["valid_evidence"].values()))
        self.assertTrue(all(result["negative_controls_rejected"].values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
