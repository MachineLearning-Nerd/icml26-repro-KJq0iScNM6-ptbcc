#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.5.1",
#   "scipy==1.16.0",
#   "matplotlib==3.10.3",
# ]
# ///
"""CPU reproduction and claim audit for the PTBCC paper.

Implementation cross-checked against the paper equations and the public
neonforestmist reproduction artifact, then rerun independently here.

The authors did not link an implementation, so this file reconstructs equations
(7)--(14) and the initialization paragraph from arXiv:2508.02123v1.  It uses
only public datasets cited in the paper and writes all evaluation evidence to
plain JSON/CSV/PNG files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import digamma, logsumexp

if __package__:
    from repro.src.data_bootstrap import ensure_public_data
    from repro.src.reference_baselines import fit_bwa, fit_fgbcc, fit_published_ds
else:
    from data_bootstrap import ensure_public_data
    from reference_baselines import fit_bwa, fit_fgbcc, fit_published_ds


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CrowdDataset:
    name: str
    task_names: list[str]
    worker_names: list[str]
    label_names: list[str]
    task: np.ndarray
    worker: np.ndarray
    label: np.ndarray
    truths: dict[int, int]
    source: str
    expected: tuple[int, int, int, int, int]

    @property
    def n_tasks(self) -> int:
        return len(self.task_names)

    @property
    def n_workers(self) -> int:
        return len(self.worker_names)

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    @property
    def n_labels(self) -> int:
        return len(self.label)

    def validate(self) -> None:
        actual = (
            self.n_tasks,
            self.n_workers,
            len(self.truths),
            self.n_classes,
            self.n_labels,
        )
        if actual != self.expected:
            raise ValueError(f"{self.name}: expected {self.expected}, observed {actual}")


@dataclass
class Fit:
    phi: np.ndarray
    prototypes: np.ndarray
    worker_weights: np.ndarray
    iterations: int
    max_delta: float


def _indices(values: Iterable[str]) -> tuple[list[str], dict[str, int]]:
    names = sorted(set(values), key=lambda value: (len(value), value))
    return names, {value: index for index, value in enumerate(names)}


def _build_dataset(
    name: str,
    rows: list[tuple[str, str, str]],
    truth_rows: dict[str, str],
    source: str,
    expected: tuple[int, int, int, int, int],
) -> CrowdDataset:
    tasks, task_index = _indices(row[0] for row in rows)
    workers, worker_index = _indices(row[1] for row in rows)
    labels, label_index = _indices(
        [row[2] for row in rows]
        + [truth for task, truth in truth_rows.items() if task in task_index]
    )
    dataset = CrowdDataset(
        name=name,
        task_names=tasks,
        worker_names=workers,
        label_names=labels,
        task=np.asarray([task_index[row[0]] for row in rows], dtype=np.int64),
        worker=np.asarray([worker_index[row[1]] for row in rows], dtype=np.int64),
        label=np.asarray([label_index[row[2]] for row in rows], dtype=np.int64),
        truths={
            task_index[task]: label_index[truth]
            for task, truth in truth_rows.items()
            if task in task_index and truth in label_index
        },
        source=source,
        expected=expected,
    )
    dataset.validate()
    return dataset


def load_active(name: str, filename: str, expected: tuple[int, int, int, int, int]) -> CrowdDataset:
    path = ROOT / "external" / "active-crowd-toolkit" / "Data" / filename
    rows: list[tuple[str, str, str]] = []
    truths: dict[str, str] = {}
    with path.open(newline="") as handle:
        for worker, task, label, truth in csv.reader(handle):
            rows.append((task, worker, label))
            truths[task] = truth
    return _build_dataset(name, rows, truths, str(path.relative_to(ROOT)), expected)


def load_split_csv(
    name: str,
    folder: str,
    expected: tuple[int, int, int, int, int],
    *,
    deduplicate: bool = False,
) -> CrowdDataset:
    base = ROOT / "external" / "crowd_truth_datasets" / "datasets" / folder
    pairs: dict[tuple[str, str], str] = {}
    rows: list[tuple[str, str, str]] = []
    with (base / "answer.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for task, worker, label in reader:
            if deduplicate:
                # The paper's 89,799 Adult labels are the 89,799 unique
                # (task, worker) pairs in the released 92,721-row file.
                pairs.setdefault((task, worker), label)
            else:
                rows.append((task, worker, label))
    if deduplicate:
        rows = [(task, worker, label) for (task, worker), label in pairs.items()]
    truths: dict[str, str] = {}
    with (base / "truth.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for task, truth in reader:
            truths[task] = truth
    source = f"{(base / 'answer.csv').relative_to(ROOT)} + truth.csv"
    return _build_dataset(name, rows, truths, source, expected)


def load_web() -> CrowdDataset:
    base = ROOT / "external" / "SpectralMethodsMeetEM" / "src"
    rows: list[tuple[str, str, str]] = []
    with (base / "web_crowd.txt").open() as handle:
        for line in handle:
            task, worker, label = line.split()
            rows.append((task, worker, label))
    truths: dict[str, str] = {}
    with (base / "web_truth.txt").open() as handle:
        for line in handle:
            task, truth = line.split()
            truths[task] = truth
    return _build_dataset(
        "Web",
        rows,
        truths,
        "external/SpectralMethodsMeetEM/src/web_{crowd,truth}.txt",
        (2665, 177, 2653, 5, 15567),
    )


def load_public_datasets() -> list[CrowdDataset]:
    ensure_public_data(ROOT)
    return [
        load_active("CF", "CF.csv", (300, 461, 300, 5, 1720)),
        load_active("MS", "MS.csv", (700, 44, 700, 10, 2945)),
        load_split_csv("Dog", "s4_Dog data", (807, 109, 807, 4, 8070)),
        load_split_csv(
            "Face", "s4_Face Sentiment Identification", (584, 27, 584, 4, 5242)
        ),
        load_split_csv(
            "Adult", "s5_AdultContent", (11040, 825, 333, 4, 89799), deduplicate=True
        ),
        load_web(),
    ]


def load_official_csv(
    name: str, folder: str, expected: tuple[int, int, int, int, int]
) -> CrowdDataset:
    """Load the cleaned files released by the FGBCC authors.

    This exact 11-dataset collection matches every count in PTBCC Table 3.
    The numeric identifiers are already contiguous, but `_build_dataset`
    independently remaps and validates them rather than trusting that fact.
    """

    base = ROOT / "external" / "CodeForFGBCC" / "datasets" / folder
    rows: list[tuple[str, str, str]] = []
    with (base / "label.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((row["item"], row["worker"], row["label"]))
    truths: dict[str, str] = {}
    with (base / "truth.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            truths[row["item"]] = row["truth"]
    return _build_dataset(
        name,
        rows,
        truths,
        f"{base.relative_to(ROOT)}/{{label,truth}}.csv",
        expected,
    )


def load_paper_datasets() -> list[CrowdDataset]:
    ensure_public_data(ROOT)
    specifications = [
        ("Val7", "valence7", (100, 38, 100, 7, 1000)),
        ("Aircr", "aircrowd6", (593, 50, 593, 6, 1588)),
        ("CF", "CF", (300, 461, 300, 5, 1720)),
        ("Fact", "fact_eval", (42624, 57, 576, 3, 214915)),
        ("MS", "MS", (700, 44, 700, 10, 2945)),
        ("Dog", "s4_Dog_data", (807, 109, 807, 4, 8070)),
        (
            "Face",
            "s4_Face_Sentiment_Identification",
            (584, 27, 584, 4, 5242),
        ),
        ("Adult", "s5_AdultContent", (11040, 825, 333, 4, 89799)),
        ("Senti", "sentiment", (98980, 1960, 1000, 5, 569274)),
        ("Val5", "valence5", (100, 38, 100, 5, 1000)),
        ("Web", "web", (2665, 177, 2653, 5, 15567)),
    ]
    return [
        load_official_csv(name, folder, expected)
        for name, folder, expected in specifications
    ]


def normalize_log_scores(log_scores: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(log_scores - logsumexp(log_scores, axis=axis, keepdims=True))


def vote_distribution(dataset: CrowdDataset) -> np.ndarray:
    counts = np.zeros((dataset.n_tasks, dataset.n_classes), dtype=np.float64)
    np.add.at(counts, (dataset.task, dataset.label), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(totals, 1.0)


def initial_prototypes(
    n_classes: int,
    n_prototypes: int,
    *,
    seed: int,
    random_extra: bool,
    f: float = 5.0,
    m: float = 1.35,
    e: float = 1.0,
) -> np.ndarray:
    if n_prototypes < 2:
        raise ValueError("PTBCC's published initialization requires at least two prototypes")
    result = np.empty((n_prototypes, n_classes, n_classes), dtype=np.float64)
    accurate_diag = f / (f + (n_classes - 1) * e)
    accurate_off = e / (f + (n_classes - 1) * e)
    adversarial_diag = e / (e + (n_classes - 1) * m)
    adversarial_off = m / (e + (n_classes - 1) * m)
    result[0, :, :] = accurate_off
    result[0, np.arange(n_classes), np.arange(n_classes)] = accurate_diag
    result[1, :, :] = adversarial_off
    result[1, np.arange(n_classes), np.arange(n_classes)] = adversarial_diag
    rng = np.random.default_rng(seed)
    for prototype in range(2, n_prototypes):
        if random_extra:
            result[prototype] = rng.dirichlet(np.ones(n_classes), size=n_classes)
        else:
            result[prototype] = 1.0 / n_classes
    return result


def fit_ptbcc(
    dataset: CrowdDataset,
    *,
    n_prototypes: int = 2,
    seed: int = 20260715,
    random_extra: bool = True,
    tolerance: float = 1e-3,
    max_iterations: int = 500,
) -> Fit:
    """Reconstruct the paper's mean-field coordinate updates.

    Arrays theta and phi correspond to q(x_ji) and q(z_i).  The empirical
    priors follow the initialization paragraph; its sums over tasks are read as
    sums over observed edges N_j, the only dimensionally valid interpretation.
    """

    t, w, y = dataset.task, dataset.worker, dataset.label
    n_tasks, n_workers, n_classes = (
        dataset.n_tasks,
        dataset.n_workers,
        dataset.n_classes,
    )
    phi = vote_distribution(dataset)
    v0 = initial_prototypes(
        n_classes, n_prototypes, seed=seed, random_extra=random_extra
    )

    theta_raw = np.empty((dataset.n_labels, n_prototypes), dtype=np.float64)
    for s in range(n_prototypes):
        theta_raw[:, s] = np.einsum("nk,nk->n", phi[t], v0[s, :, y])
    u = phi.sum(axis=0)
    beta = np.zeros((n_workers, n_prototypes), dtype=np.float64)
    for s in range(n_prototypes):
        np.add.at(beta[:, s], w, 0.4 * theta_raw[:, s])
    a = np.zeros((n_prototypes, n_classes, n_classes), dtype=np.float64)
    for s in range(n_prototypes):
        for k in range(n_classes):
            np.add.at(a[s, k], y, 0.5 * theta_raw[:, s] * phi[t, k])
    # Strictly positive concentrations prevent undefined digamma calls if a
    # class/prototype cell has no mass in a sparse dataset.
    beta = np.maximum(beta, 1e-9)
    a = np.maximum(a, 1e-9)
    theta = theta_raw / theta_raw.sum(axis=1, keepdims=True)

    delta = math.inf
    for iteration in range(1, max_iterations + 1):
        nu = u + phi.sum(axis=0)
        eta = beta.copy()
        for s in range(n_prototypes):
            np.add.at(eta[:, s], w, theta[:, s])
        mu = a.copy()
        for s in range(n_prototypes):
            for k in range(n_classes):
                np.add.at(mu[s, k], y, phi[t, k] * theta[:, s])

        elog_tau = digamma(nu) - digamma(nu.sum())
        elog_pi = digamma(eta) - digamma(eta.sum(axis=1, keepdims=True))
        elog_v = digamma(mu) - digamma(mu.sum(axis=2, keepdims=True))

        theta_score = elog_pi[w].copy()
        for s in range(n_prototypes):
            theta_score[:, s] += np.einsum(
                "nk,nk->n", phi[t], elog_v[s, :, y]
            )
        theta = normalize_log_scores(theta_score)

        phi_score = np.broadcast_to(elog_tau, (n_tasks, n_classes)).copy()
        for s in range(n_prototypes):
            edge_contribution = theta[:, s, None] * elog_v[s, :, y]
            for k in range(n_classes):
                np.add.at(phi_score[:, k], t, edge_contribution[:, k])
        phi_new = normalize_log_scores(phi_score)
        delta = float(np.max(np.abs(phi_new - phi)))
        phi = phi_new
        if delta < tolerance:
            break

    prototypes = mu / mu.sum(axis=2, keepdims=True)
    worker_weights = eta / eta.sum(axis=1, keepdims=True)
    return Fit(phi, prototypes, worker_weights, iteration, delta)


def fit_dawid_skene(
    dataset: CrowdDataset,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 200,
    smoothing: float = 1e-2,
) -> Fit:
    """Independent per-annotator confusion matrices, a transparent baseline."""

    t, w, y = dataset.task, dataset.worker, dataset.label
    phi = vote_distribution(dataset)
    n_tasks, n_workers, n_classes = (
        dataset.n_tasks,
        dataset.n_workers,
        dataset.n_classes,
    )
    delta = math.inf
    for iteration in range(1, max_iterations + 1):
        prior = (phi.sum(axis=0) + smoothing) / (n_tasks + smoothing * n_classes)
        confusion = np.full(
            (n_workers, n_classes, n_classes), smoothing, dtype=np.float64
        )
        for k in range(n_classes):
            np.add.at(confusion[:, k], (w, y), phi[t, k])
        confusion /= confusion.sum(axis=2, keepdims=True)
        score = np.broadcast_to(np.log(prior), (n_tasks, n_classes)).copy()
        for k in range(n_classes):
            np.add.at(score[:, k], t, np.log(confusion[w, k, y]))
        phi_new = normalize_log_scores(score)
        delta = float(np.max(np.abs(phi_new - phi)))
        phi = phi_new
        if delta < tolerance:
            break
    worker_quality = np.diagonal(confusion, axis1=1, axis2=2).mean(axis=1, keepdims=True)
    return Fit(phi, confusion, worker_quality, iteration, delta)


def accuracy(dataset: CrowdDataset, phi: np.ndarray) -> float:
    """Paper-faithful tie-aware accuracy.

    The BWA/FGBCC reference utility splits credit evenly across tied maxima.
    This detail is essential: its 11-dataset MV macro is 0.6986048, which
    rounds to the paper's 0.6986; first-index tie breaking does not.
    """

    task_ids = np.asarray(sorted(dataset.truths), dtype=np.int64)
    truth = np.asarray([dataset.truths[int(task)] for task in task_ids], dtype=np.int64)
    selected = phi[task_ids]
    tied = selected == selected.max(axis=1, keepdims=True)
    credit = tied[np.arange(len(task_ids)), truth] / tied.sum(axis=1)
    return float(credit.mean())


def best_permutation_distance(learned: np.ndarray, truth: np.ndarray) -> float:
    distances = np.zeros((len(learned), len(truth)), dtype=np.float64)
    for i in range(len(learned)):
        for j in range(len(truth)):
            distances[i, j] = np.mean(np.abs(learned[i] - truth[j]))
    rows, cols = linear_sum_assignment(distances)
    return float(distances[rows, cols].mean())


def synthetic_trial(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_tasks, n_workers, n_classes, n_prototypes = 800, 100, 5, 2
    v = initial_prototypes(n_classes, n_prototypes, seed=seed, random_extra=True)
    v[0] = 0.0375
    v[0, np.arange(n_classes), np.arange(n_classes)] = 0.85
    # Prototype 1 is adversarial but symmetric, matching the qualitative
    # structure of the paper's second initialization prototype.
    v[1] = 0.24
    for k in range(n_classes):
        v[1, k, k] = 0.04
    v[1] /= v[1].sum(axis=1, keepdims=True)
    good = rng.random(n_workers) < 0.65
    weights = np.vstack(
        [rng.dirichlet([18.0, 2.0] if is_good else [2.0, 18.0]) for is_good in good]
    )
    truths = rng.integers(0, n_classes, size=n_tasks)
    rows: list[tuple[str, str, str]] = []
    for task in range(n_tasks):
        for worker in rng.choice(n_workers, size=12, replace=False):
            prototype = rng.choice(n_prototypes, p=weights[worker])
            label = rng.choice(n_classes, p=v[prototype, truths[task]])
            rows.append((str(task), str(worker), str(label)))
    dataset = _build_dataset(
        f"synthetic-{seed}",
        rows,
        {str(i): str(int(z)) for i, z in enumerate(truths)},
        "generated from the PTBCC graphical model",
        (n_tasks, n_workers, n_tasks, n_classes, n_tasks * 12),
    )
    vote = vote_distribution(dataset)
    fit = fit_ptbcc(dataset, seed=seed)
    return {
        "seed": seed,
        "mv_accuracy": accuracy(dataset, vote),
        "ptbcc_accuracy": accuracy(dataset, fit.phi),
        "prototype_mae_after_matching": best_permutation_distance(fit.prototypes, v),
        "iterations": fit.iterations,
        "max_delta": fit.max_delta,
    }


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_claims() -> dict[str, object]:
    source = (ROOT / "source" / "Formatting-Instructions-LaTeX-2026.tex").read_text()
    probes = {
        "paper_method_name_ptbcc": "\\textbf{PTBCC}" in source,
        "paper_says_11_datasets": "Experiments on 11 real-world datasets" in source,
        "paper_says_15_percent": "up to 15\\% higher accuracy" in source,
        "paper_contains_26_percent": "26\\%" in source,
        "paper_contains_68_73": "0.6873" in source or "68.73" in source,
        "paper_contains_74_11": "0.7411" in source or "74.11" in source,
    }
    table4 = {"MV": 0.6986, "BWA": 0.7132, "FGBCC": 0.7175, "PTBCC": 0.7472}
    table5 = {2: 0.7472, 3: 0.7300, 4: 0.7271, 5: 0.7259, 6: 0.7229, 7: 0.7205}
    return {
        "source_sha256_note": "arXiv:2508.02123v1 source downloaded 2026-07-15",
        "text_probes": probes,
        "reported_table4": table4,
        "reported_table4_checks": {
            "ptbcc_minus_mv_percentage_points": 100 * (table4["PTBCC"] - table4["MV"]),
            "ptbcc_minus_best_baseline_percentage_points": 100
            * (table4["PTBCC"] - max(table4["BWA"], table4["FGBCC"])),
            "ptbcc_relative_gain_over_mv_percent": 100
            * (table4["PTBCC"] / table4["MV"] - 1),
        },
        "reported_table5_prototype_ablation": table5,
        "reported_table5_peak_S": max(table5, key=table5.get),
        "current_catalog_verdicts": {
            "CPBCC yields up to 26% accuracy improvement": "falsified by the paper source: method is PTBCC and best-case improvement is 15 percentage points",
            "average 68.73% to 74.11% across 10 datasets": "falsified by Table 4 and setup: MV 69.86%, PTBCC 74.72%, 11 datasets",
            "class-specific prototypes and annotator-specific weights": "verified by the generative process and equations 8, 9, 11, 13",
        },
    }


def make_figures(results: list[dict[str, object]], synthetic: list[dict[str, float]], output: Path) -> None:
    names = [str(row["dataset"]) for row in results]
    x = np.arange(len(names))
    width = 0.16
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.bar(x - 2 * width, [float(row["mv_accuracy"]) for row in results], width, label="MV")
    ax.bar(x - width, [float(row["bwa_accuracy"]) for row in results], width, label="BWA")
    ax.bar(x, [float(row["ds_accuracy"]) for row in results], width, label="Dawid-Skene")
    ax.bar(x + width, [float(row["fgbcc_accuracy"]) for row in results], width, label="FGBCC")
    ax.bar(x + 2 * width, [float(row["ptbcc_accuracy"]) for row in results], width, label="PTBCC reconstruction")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy on released ground truth")
    ax.set_title("Full 11-dataset CPU reproduction")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=5, loc="lower center")
    fig.tight_layout()
    fig.savefig(output / "public_dataset_accuracy.png", dpi=180)
    plt.close(fig)

    gains = np.asarray([row["ptbcc_accuracy"] - row["mv_accuracy"] for row in synthetic])
    maes = np.asarray([row["prototype_mae_after_matching"] for row in synthetic])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axes[0].hist(100 * gains, bins=15, color="#0f766e", edgecolor="white")
    axes[0].axvline(100 * gains.mean(), color="#b91c1c", linestyle="--", label=f"mean {100*gains.mean():.2f} pp")
    axes[0].set_xlabel("PTBCC - MV accuracy (percentage points)")
    axes[0].set_ylabel("Trials")
    axes[0].legend()
    axes[1].hist(maes, bins=15, color="#2563eb", edgecolor="white")
    axes[1].axvline(maes.mean(), color="#b91c1c", linestyle="--", label=f"mean {maes.mean():.3f}")
    axes[1].set_xlabel("Prototype MAE after Hungarian matching")
    axes[1].legend()
    fig.suptitle(f"Exact-model recovery across {len(synthetic)} deterministic seeds")
    fig.tight_layout()
    fig.savefig(output / "synthetic_recovery.png", dpi=180)
    plt.close(fig)


def _positive_timing_samples(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if float(row["wall_seconds"]) <= 0 or float(row["cpu_seconds"]) <= 0:
            raise ValueError("timing samples must be strictly positive")


def _bootstrap_median_interval(
    samples: list[float], *, seed: int = 20260723, draws: int = 10000
) -> tuple[float, float]:
    values = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    lower, upper = np.quantile(medians, [0.025, 0.975])
    return float(lower), float(upper)


def process_isolated_benchmark(
    expected_rows: list[dict[str, object]],
    *,
    repetitions: int = 5,
) -> dict[str, object]:
    """Benchmark four methods in clean processes with balanced launch order."""

    methods = ["ptbcc", "fgbcc", "ds", "bwa"]
    expected = {
        (str(row["dataset"]), method): float(row[f"{method}_accuracy"])
        for row in expected_rows
        for method in methods
    }
    raw: list[dict[str, object]] = []
    for repetition in range(repetitions):
        offset = repetition % len(methods)
        order = methods[offset:] + methods[:offset]
        for order_index, method in enumerate(order):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "repro.src.benchmark_worker",
                    "--method",
                    method,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"benchmark worker failed for {method}: {completed.stderr[-2000:]}"
                )
            marker = next(
                (
                    line.removeprefix("BENCHMARK_WORKER ")
                    for line in reversed(completed.stdout.splitlines())
                    if line.startswith("BENCHMARK_WORKER ")
                ),
                None,
            )
            if marker is None:
                raise RuntimeError(f"benchmark worker emitted no result for {method}")
            payload = json.loads(marker)
            for row in payload["rows"]:
                observed = float(row["accuracy"])
                target = expected[(str(row["dataset"]), method)]
                if not math.isclose(observed, target, rel_tol=0.0, abs_tol=1e-12):
                    raise AssertionError(
                        f"benchmark accuracy drift for {method}/{row['dataset']}: "
                        f"{observed} != {target}"
                    )
                row.update(
                    {
                        "method": method,
                        "repetition": repetition,
                        "order_index": order_index,
                    }
                )
                raw.append(row)
            print(
                "BENCHMARK_PROGRESS "
                + json.dumps(
                    {
                        "method": method,
                        "repetition": repetition,
                        "order_index": order_index,
                        "wall_total": sum(
                            float(row["wall_seconds"]) for row in payload["rows"]
                        ),
                        "cpu_total": sum(
                            float(row["cpu_seconds"]) for row in payload["rows"]
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    _positive_timing_samples(raw)

    negative = [dict(raw[0])]
    negative[0]["wall_seconds"] = 0.0
    negative_control_rejected = False
    try:
        _positive_timing_samples(negative)
    except ValueError:
        negative_control_rejected = True
    if not negative_control_rejected:
        raise AssertionError("zero-time benchmark negative control was not rejected")

    totals: list[dict[str, object]] = []
    for repetition in range(repetitions):
        for method in methods:
            selected = [
                row
                for row in raw
                if int(row["repetition"]) == repetition
                and str(row["method"]) == method
            ]
            totals.append(
                {
                    "repetition": repetition,
                    "method": method,
                    "wall_seconds": sum(
                        float(row["wall_seconds"]) for row in selected
                    ),
                    "cpu_seconds": sum(
                        float(row["cpu_seconds"]) for row in selected
                    ),
                }
            )

    ratios: list[dict[str, object]] = []
    for clock in ("wall_seconds", "cpu_seconds"):
        for comparator in ("fgbcc", "ds", "bwa"):
            samples = []
            for repetition in range(repetitions):
                numerator = next(
                    float(row[clock])
                    for row in totals
                    if row["method"] == "ptbcc"
                    and int(row["repetition"]) == repetition
                )
                denominator = next(
                    float(row[clock])
                    for row in totals
                    if row["method"] == comparator
                    and int(row["repetition"]) == repetition
                )
                samples.append(numerator / denominator)
            lower, upper = _bootstrap_median_interval(samples)
            ratios.append(
                {
                    "clock": clock,
                    "comparator": comparator,
                    "samples": samples,
                    "median": float(np.median(samples)),
                    "bootstrap_95_percent": [lower, upper],
                    "all_below_0_10": all(sample < 0.10 for sample in samples),
                }
            )
    return {
        "repetitions": repetitions,
        "process_isolated": True,
        "method_order_rotated": True,
        "raw": raw,
        "totals": totals,
        "ratios": ratios,
        "negative_control": {
            "injected_zero_time_rejected": negative_control_rejected
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "full")
    parser.add_argument("--synthetic-seeds", type=int, default=40)
    parser.add_argument("--ablation-seeds", type=int, default=3)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    datasets = load_paper_datasets()
    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    prototype_rows: list[dict[str, object]] = []
    worker_rows: list[dict[str, object]] = []
    fits: dict[str, Fit] = {}
    for dataset in datasets:
        mv_phi = vote_distribution(dataset)
        start = time.perf_counter()
        bwa = fit_bwa(dataset)
        bwa_seconds = time.perf_counter() - start
        start = time.perf_counter()
        ds = fit_published_ds(dataset)
        ds_seconds = time.perf_counter() - start
        start = time.perf_counter()
        ptbcc = fit_ptbcc(dataset)
        ptbcc_seconds = time.perf_counter() - start
        start = time.perf_counter()
        fgbcc = fit_fgbcc(dataset)
        fgbcc_seconds = time.perf_counter() - start
        fits[dataset.name] = ptbcc
        row = {
            "dataset": dataset.name,
            "tasks": dataset.n_tasks,
            "workers": dataset.n_workers,
            "truths": len(dataset.truths),
            "classes": dataset.n_classes,
            "labels": dataset.n_labels,
            "mv_accuracy": accuracy(dataset, mv_phi),
            "bwa_accuracy": accuracy(dataset, bwa.scores),
            "ds_accuracy": accuracy(dataset, ds.scores),
            "fgbcc_accuracy": accuracy(dataset, fgbcc.scores),
            "ptbcc_accuracy": accuracy(dataset, ptbcc.phi),
            "bwa_seconds": bwa_seconds,
            "ds_seconds": ds_seconds,
            "fgbcc_seconds": fgbcc_seconds,
            "ptbcc_seconds": ptbcc_seconds,
            "bwa_iterations": bwa.iterations,
            "bwa_converged": bwa.converged,
            "ds_iterations": ds.iterations,
            "fgbcc_iterations": fgbcc.iterations,
            "fgbcc_converged": fgbcc.converged,
            "ptbcc_iterations": ptbcc.iterations,
            "ptbcc_max_delta": ptbcc.max_delta,
            "source": dataset.source,
        }
        result_rows.append(row)
        print("DATASET_RESULT " + json.dumps(row, sort_keys=True), flush=True)
        for task_index in sorted(dataset.truths):
            prediction_rows.append(
                {
                    "dataset": dataset.name,
                    "task": dataset.task_names[task_index],
                    "truth": dataset.label_names[dataset.truths[task_index]],
                    "mv_prediction": dataset.label_names[int(np.argmax(mv_phi[task_index]))],
                    "bwa_prediction": dataset.label_names[int(np.argmax(bwa.scores[task_index]))],
                    "ds_prediction": dataset.label_names[int(np.argmax(ds.scores[task_index]))],
                    "fgbcc_prediction": dataset.label_names[int(np.argmax(fgbcc.scores[task_index]))],
                    "ptbcc_prediction": dataset.label_names[int(np.argmax(ptbcc.phi[task_index]))],
                    "ptbcc_confidence": float(np.max(ptbcc.phi[task_index])),
                }
            )
        for s in range(len(ptbcc.prototypes)):
            for truth_class in range(dataset.n_classes):
                for observed_class in range(dataset.n_classes):
                    prototype_rows.append(
                        {
                            "dataset": dataset.name,
                            "prototype": s,
                            "truth_class": dataset.label_names[truth_class],
                            "observed_class": dataset.label_names[observed_class],
                            "probability": ptbcc.prototypes[s, truth_class, observed_class],
                        }
                    )
        for worker_index, weights in enumerate(ptbcc.worker_weights):
            for s, weight in enumerate(weights):
                worker_rows.append(
                    {
                        "dataset": dataset.name,
                        "worker": dataset.worker_names[worker_index],
                        "prototype": s,
                        "weight": weight,
                    }
                )

    result_fields = list(result_rows[0])
    write_csv(output / "public_dataset_results.csv", result_fields, result_rows)
    write_csv(output / "ground_truth_predictions.csv", list(prediction_rows[0]), prediction_rows)
    write_csv(output / "learned_prototypes.csv", list(prototype_rows[0]), prototype_rows)
    write_csv(output / "annotator_weights.csv", list(worker_rows[0]), worker_rows)

    aircr_row = next(row for row in result_rows if row["dataset"] == "Aircr")
    fgbcc_validation = {
        "dataset": "Aircr",
        "observed_accuracy": aircr_row["fgbcc_accuracy"],
        "authors_repository_accuracy": 0.8448566610455311,
        "absolute_difference": abs(
            float(aircr_row["fgbcc_accuracy"]) - 0.8448566610455311
        ),
        "seconds": aircr_row["fgbcc_seconds"],
        "iterations": aircr_row["fgbcc_iterations"],
        "converged": aircr_row["fgbcc_converged"],
        "source_commit": "e2ca2b8a876bf9cceb871e8cec9081870a30aab4",
        "source_sha256": "6e8e1545c950c4895165eb1aa3bc37e06097e6fa7887fe9d387e1a9e7b091979",
    }
    if fgbcc_validation["absolute_difference"] > 1e-10:
        raise AssertionError(
            "Vectorized FGBCC does not reproduce the authors' Aircr golden output: "
            + json.dumps(fgbcc_validation, sort_keys=True)
        )
    (output / "fgbcc_reference_validation.json").write_text(
        json.dumps(fgbcc_validation, indent=2) + "\n"
    )
    print("FGBCC_REFERENCE_VALIDATION " + json.dumps(fgbcc_validation, sort_keys=True), flush=True)

    ablation_rows: list[dict[str, object]] = []
    for dataset in datasets:
        # Reuse the deterministic main fit for S=2.
        ablation_rows.append(
            {
                "dataset": dataset.name,
                "S": 2,
                "extra_init": "none",
                "seed": 20260715,
                "accuracy": accuracy(dataset, fits[dataset.name].phi),
                "iterations": fits[dataset.name].iterations,
            }
        )
        uniform_fit = fit_ptbcc(dataset, n_prototypes=3, random_extra=False)
        ablation_rows.append(
            {
                "dataset": dataset.name,
                "S": 3,
                "extra_init": "uniform_matrix",
                "seed": 20260715,
                "accuracy": accuracy(dataset, uniform_fit.phi),
                "iterations": uniform_fit.iterations,
            }
        )
        for n_prototypes in (3, 4):
            for seed_offset in range(args.ablation_seeds):
                seed = 20260715 + seed_offset
                fit = fit_ptbcc(dataset, n_prototypes=n_prototypes, seed=seed)
                ablation_rows.append(
                    {
                        "dataset": dataset.name,
                        "S": n_prototypes,
                        "extra_init": "Dirichlet(1)",
                        "seed": seed,
                        "accuracy": accuracy(dataset, fit.phi),
                        "iterations": fit.iterations,
                    }
                )
    write_csv(output / "prototype_count_ablation.csv", list(ablation_rows[0]), ablation_rows)
    print("ABLATION_RAW " + json.dumps(ablation_rows, sort_keys=True), flush=True)

    synthetic_rows = [synthetic_trial(73000 + seed) for seed in range(args.synthetic_seeds)]
    write_csv(output / "synthetic_recovery_trials.csv", list(synthetic_rows[0]), synthetic_rows)
    print("SYNTHETIC_RAW " + json.dumps(synthetic_rows, sort_keys=True), flush=True)
    make_figures(result_rows, synthetic_rows, output)

    claim_audit = audit_claims()
    (output / "claim_audit.json").write_text(json.dumps(claim_audit, indent=2) + "\n")
    mean_public = {
        method: float(np.mean([row[f"{method}_accuracy"] for row in result_rows]))
        for method in ("mv", "bwa", "ds", "fgbcc", "ptbcc")
    }
    ablation_macro: list[dict[str, object]] = []
    ablation_keys = sorted(
        {
            (int(row["S"]), str(row["extra_init"]), int(row["seed"]))
            for row in ablation_rows
        }
    )
    for prototypes, extra_init, seed in ablation_keys:
        selected = [
            float(row["accuracy"])
            for row in ablation_rows
            if int(row["S"]) == prototypes
            and str(row["extra_init"]) == extra_init
            and int(row["seed"]) == seed
        ]
        ablation_macro.append(
            {
                "S": prototypes,
                "extra_init": extra_init,
                "seed": seed,
                "datasets": len(selected),
                "macro_accuracy": float(np.mean(selected)),
            }
        )
    val5_row = next(row for row in result_rows if row["dataset"] == "Val5")
    best_val5_baseline = max(
        float(val5_row[f"{method}_accuracy"])
        for method in ("mv", "bwa", "ds", "fgbcc")
    )
    runtime_totals = {
        method: float(sum(float(row[f"{method}_seconds"]) for row in result_rows))
        for method in ("bwa", "ds", "fgbcc", "ptbcc")
    }
    paper_table4 = {
        "MV": 0.6986,
        "BWA": 0.7132,
        "FGBCC": 0.7175,
        "PTBCC": 0.7472,
    }
    claim_results = {
        "claim_2": {
            "ptbcc_val5": float(val5_row["ptbcc_accuracy"]),
            "best_baseline_val5": best_val5_baseline,
            "gain_percentage_points": 100.0
            * (float(val5_row["ptbcc_accuracy"]) - best_val5_baseline),
            "rounded_contract_match": (
                round(float(val5_row["ptbcc_accuracy"]), 2) == 0.56
                and round(best_val5_baseline, 2) == 0.41
            ),
        },
        "claim_3": {
            "observed": {
                method.upper(): mean_public[method]
                for method in ("mv", "bwa", "fgbcc", "ptbcc")
            },
            "paper": paper_table4,
            "all_four_decimal_matches": all(
                round(mean_public[method.lower()], 4) == expected
                for method, expected in paper_table4.items()
            ),
        },
        "claim_4": {
            "macro_by_configuration": ablation_macro,
            "paper": {"S2": 0.7472, "S3": 0.7300, "S4": 0.7271},
        },
        "claim_5_preliminary_single_sample": {
            "runtime_totals_seconds": runtime_totals,
            "ptbcc_over_fgbcc": runtime_totals["ptbcc"]
            / runtime_totals["fgbcc"],
            "ptbcc_over_ds": runtime_totals["ptbcc"] / runtime_totals["ds"],
            "ptbcc_minus_fgbcc_accuracy_pp": 100.0
            * (mean_public["ptbcc"] - mean_public["fgbcc"]),
            "controlled_repeats_still_required": True,
        },
    }
    benchmark = process_isolated_benchmark(result_rows, repetitions=5)
    (output / "runtime_benchmark.json").write_text(
        json.dumps(benchmark, indent=2) + "\n"
    )
    claim_results["claim_5_controlled"] = {
        "accuracy_gain_over_fgbcc_pp": 100.0
        * (mean_public["ptbcc"] - mean_public["fgbcc"]),
        "ratios": benchmark["ratios"],
        "negative_control": benchmark["negative_control"],
    }
    print("ABLATION_MACRO " + json.dumps(ablation_macro, sort_keys=True), flush=True)
    print("RUNTIME_BENCHMARK " + json.dumps(benchmark, sort_keys=True), flush=True)
    print("CLAIM_RESULTS " + json.dumps(claim_results, sort_keys=True), flush=True)
    summary = {
        "paper": {
            "openreview_id": "KJq0iScNM6",
            "arxiv_id": "2508.02123",
            "title_in_arxiv_v1": "Understanding the Essence: Delving into Annotator Prototype Learning for Multi-Class Annotation Aggregation",
            "title_in_challenge_index": "Let the Prototype Guide You: Robust Aggregation of Sparse Multi-Class Annotations via Annotator Prototype Learning",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "compute": "launched through OpenResearch on local CPU; no GPU",
        },
        "scope": {
            "public_datasets_reproduced": [dataset.name for dataset in datasets],
            "paper_dataset_count": 11,
            "unavailable_in_this_reproduction": [],
            "corpus_source": "FGBCC authors' cleaned release at commit e2ca2b8a876bf9cceb871e8cec9081870a30aab4",
            "corpus_matches_table3": True,
            "fgbcc_scope": "all 11 datasets after exact Aircr golden-output equivalence",
        },
        "public_dataset_macro_accuracy": mean_public,
        "public_dataset_ptbcc_minus_mv_pp": 100 * (mean_public["ptbcc"] - mean_public["mv"]),
        "synthetic": {
            "trials": len(synthetic_rows),
            "ptbcc_wins_vs_mv": sum(row["ptbcc_accuracy"] > row["mv_accuracy"] for row in synthetic_rows),
            "mean_mv_accuracy": float(np.mean([row["mv_accuracy"] for row in synthetic_rows])),
            "mean_ptbcc_accuracy": float(np.mean([row["ptbcc_accuracy"] for row in synthetic_rows])),
            "mean_prototype_mae": float(
                np.mean([row["prototype_mae_after_matching"] for row in synthetic_rows])
            ),
        },
        "claim_audit": claim_audit,
        "fgbcc_reference_validation": fgbcc_validation,
        "claim_results": claim_results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    checker = subprocess.run(
        [
            sys.executable,
            "-m",
            "repro.src.claim_verifier",
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if checker.stdout:
        print(checker.stdout, end="", flush=True)
    if checker.stderr:
        print(checker.stderr, file=sys.stderr, end="", flush=True)
    if checker.returncode != 0:
        raise RuntimeError(
            f"independent claim verifier exited {checker.returncode}"
        )


if __name__ == "__main__":
    main()
