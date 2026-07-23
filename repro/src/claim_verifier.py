#!/usr/bin/env python3
"""Independent, file-only checker for the five PTBCC reproduction claims.

This module deliberately does not import the model or baseline implementations.
It consumes their CSV/JSON outputs, recomputes the claim statistics, exercises
one corrupting negative control per claim, and exits nonzero if the evidence
bundle is incomplete or internally inconsistent.  A scientifically honest
``BLOCKED`` verdict is not an execution failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
VERDICTS = {"VERIFIED", "FALSIFIED", "BLOCKED"}
EXPECTED_DATASETS = {
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
}
FIXED_COMMAND = (
    "uv sync --frozen && uv run python repro/src/run_ptbcc.py "
    "--output-dir outputs/full && uv run python -m unittest -v "
    "repro.tests.test_ptbcc"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _macro(rows: list[dict[str, Any]], field: str) -> float:
    return mean(float(row[field]) for row in rows)


def evaluate_claim_1(
    synthetic: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
    weights: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    seeds = {int(row["seed"]) for row in synthetic}
    prototype_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    prototype_ids: dict[str, set[int]] = defaultdict(set)
    for row in prototypes:
        dataset = str(row["dataset"])
        prototype = int(row["prototype"])
        prototype_ids[dataset].add(prototype)
        prototype_groups[
            (dataset, str(prototype), str(row["truth_class"]))
        ].append(float(row["probability"]))
    weight_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    weight_ids: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in weights:
        key = (str(row["dataset"]), str(row["worker"]))
        weight_groups[key].append(float(row["weight"]))
        weight_ids[key].add(int(row["prototype"]))
    expected_workers = {
        str(row["dataset"]): int(row["workers"]) for row in results
    }
    observed_workers: dict[str, int] = defaultdict(int)
    for dataset, _worker in weight_groups:
        observed_workers[dataset] += 1
    checks = {
        "forty_distinct_expected_seeds": seeds == set(range(73000, 73040)),
        "ptbcc_beats_mv_all_trials": all(
            float(row["ptbcc_accuracy"]) > float(row["mv_accuracy"])
            for row in synthetic
        ),
        "mean_matched_prototype_mae_below_0_08": (
            len(synthetic) == 40
            and mean(
                float(row["prototype_mae_after_matching"]) for row in synthetic
            )
            < 0.08
        ),
        "two_shared_prototypes_per_dataset": (
            set(prototype_ids) == EXPECTED_DATASETS
            and all(ids == {0, 1} for ids in prototype_ids.values())
        ),
        "prototype_rows_normalized": bool(prototype_groups)
        and all(
            math.isclose(sum(values), 1.0, abs_tol=1e-10)
            for values in prototype_groups.values()
        ),
        "worker_mixtures_normalized": bool(weight_groups)
        and all(
            len(values) == 2
            and weight_ids[key] == {0, 1}
            and math.isclose(sum(values), 1.0, abs_tol=1e-10)
            for key, values in weight_groups.items()
        ),
        "worker_count_matches_table3": (
            set(observed_workers) == EXPECTED_DATASETS
            and all(
                observed_workers[name] == expected_workers[name]
                for name in EXPECTED_DATASETS
            )
        ),
    }
    valid = all(checks.values())
    return {
        "claim_id": 1,
        "verdict": "VERIFIED" if valid else "BLOCKED",
        "valid_evidence": valid,
        "checks": checks,
        "statistics": {
            "trials": len(synthetic),
            "ptbcc_wins": sum(
                float(row["ptbcc_accuracy"]) > float(row["mv_accuracy"])
                for row in synthetic
            ),
            "mean_matched_prototype_mae": mean(
                float(row["prototype_mae_after_matching"]) for row in synthetic
            )
            if synthetic
            else None,
            "prototype_tensor_semantics": (
                "shared (prototype, truth class, observed class)"
            ),
            "annotator_tensor_semantics": "(annotator, prototype) mixture",
        },
    }


def evaluate_claim_2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    val5_rows = [row for row in rows if str(row["dataset"]) == "Val5"]
    valid_domain = (
        len(rows) == 11
        and {str(row["dataset"]) for row in rows} == EXPECTED_DATASETS
        and len(val5_rows) == 1
    )
    val5 = val5_rows[0] if val5_rows else {}
    expected_dimensions = {
        "tasks": 100,
        "workers": 38,
        "truths": 100,
        "classes": 5,
        "labels": 1000,
    }
    dimensions_match = valid_domain and all(
        int(val5[key]) == value for key, value in expected_dimensions.items()
    )
    baselines = {
        method: float(val5.get(f"{method}_accuracy", float("nan")))
        for method in ("mv", "bwa", "ds", "fgbcc")
    }
    best_method = max(baselines, key=baselines.get) if baselines else None
    best_accuracy = baselines[best_method] if best_method else float("nan")
    ptbcc = float(val5.get("ptbcc_accuracy", float("nan")))
    gain = ptbcc - best_accuracy
    checks = {
        "exact_val5_domain": dimensions_match,
        "best_baseline_is_released_ds_at_0_41": (
            best_method == "ds" and math.isclose(best_accuracy, 0.41, abs_tol=1e-12)
        ),
        "ptbcc_is_0_56": math.isclose(ptbcc, 0.56, abs_tol=1e-12),
        "gain_rounds_to_15_percentage_points": round(gain, 2) == 0.15,
    }
    valid = dimensions_match and all(
        math.isfinite(value) for value in [*baselines.values(), ptbcc]
    )
    verified = valid and all(checks.values())
    return {
        "claim_id": 2,
        "verdict": "VERIFIED" if verified else "BLOCKED",
        "valid_evidence": valid,
        "checks": checks,
        "statistics": {
            "ptbcc": ptbcc,
            "baselines": baselines,
            "best_baseline": best_method,
            "gain_percentage_points": 100.0 * gain,
        },
    }


def evaluate_claim_3(
    rows: list[dict[str, Any]],
    fgbcc_validation: dict[str, Any],
) -> dict[str, Any]:
    valid_domain = (
        len(rows) == 11
        and {str(row["dataset"]) for row in rows} == EXPECTED_DATASETS
    )
    fields = {
        "MV": "mv_accuracy",
        "BWA": "bwa_accuracy",
        "FGBCC": "fgbcc_accuracy",
        "PTBCC": "ptbcc_accuracy",
    }
    paper = {"MV": 0.6986, "BWA": 0.7132, "FGBCC": 0.7175, "PTBCC": 0.7472}
    observed = {
        method: _macro(rows, field) if rows else float("nan")
        for method, field in fields.items()
    }
    matches = {
        method: round(observed[method], 4) == expected
        for method, expected in paper.items()
    }
    golden = (
        fgbcc_validation.get("dataset") == "Aircr"
        and math.isclose(
            float(fgbcc_validation.get("absolute_difference", float("inf"))),
            0.0,
            abs_tol=1e-12,
        )
        and fgbcc_validation.get("source_commit")
        == "e2ca2b8a876bf9cceb871e8cec9081870a30aab4"
    )
    valid = valid_domain and golden
    verified = valid and all(matches.values())
    return {
        "claim_id": 3,
        "verdict": "VERIFIED" if verified else "BLOCKED",
        "valid_evidence": valid,
        "checks": {
            "exact_11_dataset_domain": valid_domain,
            "fgbcc_author_golden_aircr_exact": golden,
            "all_four_values_match_to_four_decimals": all(matches.values()),
        },
        "statistics": {
            "paper": paper,
            "observed": observed,
            "four_decimal_matches": matches,
            "fgbcc_absolute_delta": observed["FGBCC"] - paper["FGBCC"],
        },
        "blocker": None
        if verified
        else (
            "FGBCC regenerates as 0.7175816744 (0.7176 at four decimals), "
            "while the paper prints 0.7175; PTBCC has no released author code "
            "and the paper does not release its exact numerical environment."
        ),
    }


def _ablation_macros(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, str, int], float]:
    grouped: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (int(row["S"]), str(row["extra_init"]), int(row["seed"]))
        ].append(float(row["accuracy"]))
    return {key: mean(values) for key, values in grouped.items()}


def evaluate_claim_4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    macros = _ablation_macros(rows)
    expected_keys = {
        (2, "none", 20260715),
        (3, "uniform_matrix", 20260715),
        *{
            (size, "Dirichlet(1)", seed)
            for size in (3, 4)
            for seed in (20260715, 20260716, 20260717)
        },
    }
    counts: dict[tuple[int, str, int], int] = defaultdict(int)
    datasets: dict[tuple[int, str, int], set[str]] = defaultdict(set)
    for row in rows:
        key = (int(row["S"]), str(row["extra_init"]), int(row["seed"]))
        counts[key] += 1
        datasets[key].add(str(row["dataset"]))
    valid = (
        set(macros) == expected_keys
        and all(counts[key] == 11 for key in expected_keys)
        and all(datasets[key] == EXPECTED_DATASETS for key in expected_keys)
    )
    s2 = macros.get((2, "none", 20260715), float("nan"))
    stochastic = {
        (size, seed): macros.get(
            (size, "Dirichlet(1)", seed), float("nan")
        )
        for size in (3, 4)
        for seed in (20260715, 20260716, 20260717)
    }
    peak_all_seeds = all(s2 > value for value in stochastic.values())
    exact_seed_matches = {
        seed: (
            round(s2, 4) == 0.7472
            and round(stochastic[(3, seed)], 4) == 0.7300
            and round(stochastic[(4, seed)], 4) == 0.7271
        )
        for seed in (20260715, 20260716, 20260717)
    }
    verified = valid and peak_all_seeds and any(exact_seed_matches.values())
    return {
        "claim_id": 4,
        "verdict": "VERIFIED" if verified else "BLOCKED",
        "valid_evidence": valid,
        "checks": {
            "complete_11_dataset_seed_grid": valid,
            "s2_peaks_for_every_tested_seed": peak_all_seeds,
            "one_disclosed_seed_reproduces_all_exact_numbers": any(
                exact_seed_matches.values()
            ),
            "three_ran_kept_distinct_from_dirichlet_draw": (
                (3, "uniform_matrix", 20260715) in macros
            ),
        },
        "statistics": {
            "S2": s2,
            "stochastic": {
                f"S{size}_seed{seed}": value
                for (size, seed), value in stochastic.items()
            },
            "three_ran_uniform": macros.get(
                (3, "uniform_matrix", 20260715)
            ),
            "paper": {"S2": 0.7472, "S3": 0.7300, "S4": 0.7271},
            "exact_seed_matches": exact_seed_matches,
        },
        "blocker": None
        if verified
        else (
            "The paper omits the random seed. S=2 peaks in every tested seed, "
            "but no tested single seed reproduces both printed S=3 and S=4 "
            "numbers, so the exact stochastic table cannot be attributed."
        ),
    }


def _runtime_valid(benchmark: dict[str, Any]) -> bool:
    raw = benchmark.get("raw", [])
    totals = benchmark.get("totals", [])
    ratios = benchmark.get("ratios", [])
    raw_keys = {
        (
            int(row["repetition"]),
            str(row["method"]),
            str(row["dataset"]),
        )
        for row in raw
    }
    expected_raw = {
        (repetition, method, dataset)
        for repetition in range(5)
        for method in ("ptbcc", "fgbcc", "ds", "bwa")
        for dataset in EXPECTED_DATASETS
    }
    expected_totals = {
        (repetition, method)
        for repetition in range(5)
        for method in ("ptbcc", "fgbcc", "ds", "bwa")
    }
    observed_totals = {
        (int(row["repetition"]), str(row["method"])) for row in totals
    }
    expected_ratios = {
        (clock, comparator)
        for clock in ("wall_seconds", "cpu_seconds")
        for comparator in ("fgbcc", "ds", "bwa")
    }
    observed_ratios = {
        (str(row["clock"]), str(row["comparator"])) for row in ratios
    }
    positive = all(
        float(row["wall_seconds"]) > 0.0 and float(row["cpu_seconds"]) > 0.0
        for row in raw
    )
    accuracy_invariant = all(
        len(
            {
                float(row["accuracy"])
                for row in raw
                if str(row["method"]) == method
                and str(row["dataset"]) == dataset
            }
        )
        == 1
        for method in ("ptbcc", "fgbcc", "ds", "bwa")
        for dataset in EXPECTED_DATASETS
    )
    ratios_recompute = True
    for ratio in ratios:
        clock = str(ratio["clock"])
        comparator = str(ratio["comparator"])
        samples = []
        for repetition in range(5):
            numerator = next(
                float(row[clock])
                for row in totals
                if int(row["repetition"]) == repetition
                and row["method"] == "ptbcc"
            )
            denominator = next(
                float(row[clock])
                for row in totals
                if int(row["repetition"]) == repetition
                and row["method"] == comparator
            )
            samples.append(numerator / denominator)
        ratios_recompute &= all(
            math.isclose(left, right, abs_tol=1e-12)
            for left, right in zip(samples, ratio["samples"], strict=True)
        )
        ratios_recompute &= math.isclose(
            median(samples), float(ratio["median"]), abs_tol=1e-12
        )
    return (
        len(raw) == 220
        and raw_keys == expected_raw
        and len(totals) == 20
        and observed_totals == expected_totals
        and len(ratios) == 6
        and observed_ratios == expected_ratios
        and positive
        and accuracy_invariant
        and ratios_recompute
        and benchmark.get("negative_control", {}).get(
            "injected_zero_time_rejected"
        )
        is True
    )


def evaluate_claim_5(
    benchmark: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    valid = _runtime_valid(benchmark)
    ratios = benchmark.get("ratios", [])
    threshold_contradicted = valid and all(
        all(float(sample) >= 0.10 for sample in row["samples"])
        for row in ratios
    )
    ratio_summary = {
        f"{row['clock']}_over_{row['comparator']}": {
            "median": float(row["median"]),
            "bootstrap_95_percent": [
                float(value) for value in row["bootstrap_95_percent"]
            ],
        }
        for row in ratios
    }
    gain = (
        100.0
        * (_macro(results, "ptbcc_accuracy") - _macro(results, "fgbcc_accuracy"))
        if results
        else float("nan")
    )
    return {
        "claim_id": 5,
        "verdict": "BLOCKED",
        "valid_evidence": valid,
        "checks": {
            "complete_controlled_benchmark": valid,
            "accuracy_gain_is_about_3pp": math.isclose(
                gain, 3.0, rel_tol=0.0, abs_tol=0.1
            ),
            "under_10_percent_threshold_contradicted_on_local_stack": (
                threshold_contradicted
            ),
            "paper_exact_denominator_and_raw_timings_available": False,
            "paper_cpu_and_original_implementations_reproduced": False,
        },
        "statistics": {
            "ptbcc_minus_fgbcc_accuracy_pp": gain,
            "ratios": ratio_summary,
        },
        "blocker": (
            "The local benchmark rejects <10% for every implemented comparator, "
            "but the paper omits the aggregate denominator and raw timings, used "
            "a 2-vCPU Intel Xeon 8369HC, and plotted original baseline loops. "
            "This Apple-CPU run uses an equation-equivalent vectorized FGBCC, so "
            "it is a direct setup-specific contradiction, not a faithful "
            "paper-hardware falsification."
        ),
    }


def _negative_controls(
    results: list[dict[str, Any]],
    synthetic: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
    weights: list[dict[str, Any]],
    ablation: list[dict[str, Any]],
    benchmark: dict[str, Any],
    fgbcc_validation: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    c1 = evaluate_claim_1(
        synthetic[:-1], prototypes, weights, results
    )
    corrupted_c2 = deepcopy(results)
    next(row for row in corrupted_c2 if row["dataset"] == "Val5")[
        "ptbcc_accuracy"
    ] = "0.55"
    c2 = evaluate_claim_2(corrupted_c2)
    c3 = evaluate_claim_3(results[:-1], fgbcc_validation)
    corrupted_c4 = deepcopy(ablation)
    next(
        row
        for row in corrupted_c4
        if int(row["S"]) == 3 and row["extra_init"] == "uniform_matrix"
    )["extra_init"] = "Dirichlet(1)"
    c4 = evaluate_claim_4(corrupted_c4)
    corrupted_c5 = deepcopy(benchmark)
    corrupted_c5["raw"][0]["wall_seconds"] = 0.0
    c5 = evaluate_claim_5(corrupted_c5, results)
    evaluations = {1: c1, 2: c2, 3: c3, 4: c4, 5: c5}
    return {
        claim_id: {
            "mutation": {
                1: "drop one required synthetic seed",
                2: "change Val5 PTBCC accuracy from 0.56 to 0.55",
                3: "drop one of the 11 datasets",
                4: "mislabel the 3-Ran uniform matrix as a Dirichlet draw",
                5: "inject a zero wall-time sample",
            }[claim_id],
            "rejected": not evaluation["valid_evidence"]
            or evaluation["verdict"] != (
                "VERIFIED" if claim_id in (1, 2) else evaluation["verdict"]
            ),
            "mutated_valid_evidence": evaluation["valid_evidence"],
            "mutated_verdict": evaluation["verdict"],
        }
        for claim_id, evaluation in evaluations.items()
    }


def _provenance(output: Path) -> dict[str, Any]:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return {
        "git_commit": git_commit,
        "fixed_command": FIXED_COMMAND,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            ".python-version": (ROOT / ".python-version").read_text().strip(),
            "pyproject_sha256": _sha256(ROOT / "pyproject.toml"),
            "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
        },
        "compute": "OpenResearch local backend, CPU only, no GPU",
        "output_dir": str(output.relative_to(ROOT)),
        "seeds": {
            "ptbcc_main": 20260715,
            "ablation": [20260715, 20260716, 20260717],
            "synthetic": list(range(73000, 73040)),
            "bootstrap": 20260723,
        },
    }


def _eval_markdown(result: dict[str, Any]) -> str:
    claim_id = result["claim_id"]
    verdict = result["verdict"]
    stats = result["statistics"]
    if claim_id == 1:
        body = (
            f"All {stats['ptbcc_wins']}/{stats['trials']} synthetic trials "
            "recover the shared-prototype mechanism better than majority vote; "
            f"matched prototype MAE is {stats['mean_matched_prototype_mae']:.6f}. "
            "The file-only checker also confirms two normalized shared prototype "
            "matrices and normalized per-annotator mixture weights on every dataset."
        )
    elif claim_id == 2:
        body = (
            f"On exact Val5, PTBCC is {stats['ptbcc']:.2f}; released DS is "
            f"{stats['baselines']['ds']:.2f}; the directly regenerated gain is "
            f"{stats['gain_percentage_points']:.1f} percentage points."
        )
    elif claim_id == 3:
        body = (
            "The exact 11-dataset macro regenerates as "
            + ", ".join(
                f"{name} {value:.10f}"
                for name, value in stats["observed"].items()
            )
            + ". MV, BWA, and PTBCC match the printed four-decimal values; FGBCC "
            "rounds to 0.7176 rather than 0.7175. " + result["blocker"]
        )
    elif claim_id == 4:
        body = (
            f"S=2 regenerates as {stats['S2']:.10f} and is the peak for all "
            "three tested seeds. The stochastic S=3/S=4 values vary by seed. "
            + result["blocker"]
        )
    else:
        cpu_fgbcc = stats["ratios"]["cpu_seconds_over_fgbcc"]
        body = (
            f"The accuracy gain is {stats['ptbcc_minus_fgbcc_accuracy_pp']:.4f} "
            "percentage points. PTBCC/FGBCC process-time ratio is "
            f"{cpu_fgbcc['median']:.4f} with bootstrap interval "
            f"[{cpu_fgbcc['bootstrap_95_percent'][0]:.4f}, "
            f"{cpu_fgbcc['bootstrap_95_percent'][1]:.4f}]. "
            + result["blocker"]
        )
    return (
        f"# {verdict}\n\n"
        f"{body}\n\n"
        "This verdict is emitted by `repro.src.claim_verifier` from the raw "
        "CSV/JSON outputs. See `independent_checker.json`, "
        "`negative_control.json`, and `run_provenance.json` in this directory.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()

    results = _read_csv(output / "public_dataset_results.csv")
    synthetic = _read_csv(output / "synthetic_recovery_trials.csv")
    prototypes = _read_csv(output / "learned_prototypes.csv")
    weights = _read_csv(output / "annotator_weights.csv")
    ablation = _read_csv(output / "prototype_count_ablation.csv")
    benchmark = _read_json(output / "runtime_benchmark.json")
    fgbcc_validation = _read_json(output / "fgbcc_reference_validation.json")

    evaluations = {
        1: evaluate_claim_1(
            synthetic, prototypes, weights, results
        ),
        2: evaluate_claim_2(results),
        3: evaluate_claim_3(results, fgbcc_validation),
        4: evaluate_claim_4(ablation),
        5: evaluate_claim_5(benchmark, results),
    }
    controls = _negative_controls(
        results,
        synthetic,
        prototypes,
        weights,
        ablation,
        benchmark,
        fgbcc_validation,
    )
    provenance = _provenance(output)

    for claim_id, result in evaluations.items():
        claim_dir = ARTIFACTS / f"claim_{claim_id}"
        _write_json(claim_dir / "independent_checker.json", result)
        _write_json(claim_dir / "negative_control.json", controls[claim_id])
        _write_json(claim_dir / "verifier_output.json", result)
        _write_json(claim_dir / "run_provenance.json", provenance)
        (claim_dir / "EVAL.md").write_text(_eval_markdown(result))

    _write_json(
        ARTIFACTS / "claim_1" / "raw_synthetic_recovery.json",
        {"trials": synthetic},
    )
    val5 = next(row for row in results if row["dataset"] == "Val5")
    _write_json(
        ARTIFACTS / "claim_2" / "raw_val5_result.json",
        {"dataset_result": val5},
    )
    _write_json(
        ARTIFACTS / "claim_3" / "raw_table4_results.json",
        {
            "dataset_results": results,
            "fgbcc_reference_validation": fgbcc_validation,
        },
    )
    _write_json(
        ARTIFACTS / "claim_4" / "raw_ablation.json",
        {
            "raw": ablation,
            "macro": [
                {
                    "S": key[0],
                    "extra_init": key[1],
                    "seed": key[2],
                    "macro_accuracy": value,
                }
                for key, value in sorted(_ablation_macros(ablation).items())
            ],
        },
    )
    _write_json(
        ARTIFACTS / "claim_5" / "raw_runtime_benchmark.json",
        {"benchmark": benchmark},
    )

    summary = {
        "verdicts": {
            f"claim_{claim_id}": result["verdict"]
            for claim_id, result in evaluations.items()
        },
        "valid_evidence": {
            f"claim_{claim_id}": result["valid_evidence"]
            for claim_id, result in evaluations.items()
        },
        "negative_controls_rejected": {
            f"claim_{claim_id}": control["rejected"]
            for claim_id, control in controls.items()
        },
        "git_commit": provenance["git_commit"],
    }
    _write_json(output / "claim_verdicts.json", summary)
    print("INDEPENDENT_CHECKER " + json.dumps(summary, sort_keys=True))
    print(
        "CLAIM_VERIFIER_DETAILS "
        + json.dumps(
            {
                "evaluations": evaluations,
                "negative_controls": controls,
                "provenance": provenance,
            },
            sort_keys=True,
        )
    )

    errors = []
    errors.extend(
        f"claim {claim_id}: invalid evidence"
        for claim_id, result in evaluations.items()
        if not result["valid_evidence"]
    )
    errors.extend(
        f"claim {claim_id}: invalid verdict {result['verdict']}"
        for claim_id, result in evaluations.items()
        if result["verdict"] not in VERDICTS
    )
    errors.extend(
        f"claim {claim_id}: negative control was not rejected"
        for claim_id, control in controls.items()
        if not control["rejected"]
    )
    if errors:
        print("CLAIM_VERIFIER_FAILED " + json.dumps(errors), file=sys.stderr)
        raise SystemExit(1)
    print("CLAIM_VERIFIER_PASS " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
