#!/usr/bin/env python3
"""Build the evidence figures used by the PTBCC visual reproduction report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COLORS = {
    "MV": "#8c8c8c",
    "BWA": "#4c78a8",
    "DS": "#f58518",
    "FGBCC": "#54a24b",
    "PTBCC": "#b279a2",
    "S2": "#b279a2",
    "S3": "#4c78a8",
    "S4": "#f58518",
}


def _read(path: Path) -> object:
    return json.loads(path.read_text())


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.18,
            "font.size": 10,
            "figure.titlesize": 15,
        }
    )


def val5_figure(artifacts: Path, images: Path) -> None:
    payload = _read(artifacts / "claim_2" / "raw_val5_result.json")
    row = payload["dataset_result"]
    labels = ["MV", "BWA", "FGBCC", "DS", "PTBCC"]
    values = [
        float(row["mv_accuracy"]),
        float(row["bwa_accuracy"]),
        float(row["fgbcc_accuracy"]),
        float(row["ds_accuracy"]),
        float(row["ptbcc_accuracy"]),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    bars = ax.bar(
        labels,
        values,
        color=[COLORS[label] for label in labels],
        width=0.68,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}" if value not in (0.41, 0.56) else f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold" if value in (0.41, 0.56) else "normal",
        )
    ax.annotate(
        "+15.0 percentage points",
        xy=(4, 0.56),
        xytext=(3, 0.66),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Fractional-tie accuracy")
    ax.set_title("Exact Val5 reproduction: PTBCC vs faithful baselines", loc="left")
    ax.text(
        0.01,
        -0.19,
        "Exact corpus: 100 tasks · 38 annotators · 5 classes · 1,000 labels",
        transform=ax.transAxes,
        color="#555555",
    )
    _save(fig, images / "headline-val5.png")


def table4_figure(artifacts: Path, images: Path) -> None:
    payload = _read(artifacts / "claim_3" / "raw_table4_results.json")
    rows = payload["dataset_results"]
    methods = ["MV", "BWA", "FGBCC", "PTBCC"]
    fields = {
        "MV": "mv_accuracy",
        "BWA": "bwa_accuracy",
        "FGBCC": "fgbcc_accuracy",
        "PTBCC": "ptbcc_accuracy",
    }
    paper = {"MV": 0.6986, "BWA": 0.7132, "FGBCC": 0.7175, "PTBCC": 0.7472}
    observed = {
        method: mean(float(row[fields[method]]) for row in rows)
        for method in methods
    }
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    width = 0.34
    paper_bars = ax.bar(
        x - width / 2,
        [paper[m] for m in methods],
        width,
        color="#d9d9d9",
        edgecolor="#777777",
        label="Paper Table 4",
    )
    observed_bars = ax.bar(
        x + width / 2,
        [observed[m] for m in methods],
        width,
        color=[COLORS[m] for m in methods],
        label="Regenerated",
    )
    for bars in (paper_bars, observed_bars):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=35,
            )
    ax.set_xticks(x, methods)
    ax.set_ylim(0.66, 0.77)
    ax.set_ylabel("11-dataset macro accuracy")
    ax.set_title("Table 4: three exact four-decimal matches, one tiny mismatch", loc="left")
    ax.legend(frameon=False, ncols=2)
    ax.text(
        0.01,
        -0.20,
        "FGBCC regenerates as 0.7175817 → 0.7176, while the paper prints 0.7175.",
        transform=ax.transAxes,
        color="#7f3b08",
    )
    _save(fig, images / "table4-macro-accuracy.png")


def ablation_figure(artifacts: Path, images: Path) -> None:
    payload = _read(artifacts / "claim_4" / "raw_ablation.json")
    rows = payload["raw"]
    grouped: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (int(row["S"]), str(row["extra_init"]), int(row["seed"]))
        ].append(float(row["accuracy"]))
    macro = {key: mean(values) for key, values in grouped.items()}
    seeds = [20260715, 20260716, 20260717]
    s2 = macro[(2, "none", 20260715)]
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for seed_index, seed in enumerate(seeds):
        values = [
            s2,
            macro[(3, "Dirichlet(1)", seed)],
            macro[(4, "Dirichlet(1)", seed)],
        ]
        ax.plot(
            [2, 3, 4],
            values,
            marker="o",
            linewidth=1.8,
            alpha=0.85,
            label=f"seed {seed}",
        )
    paper_values = [0.7472, 0.7300, 0.7271]
    ax.scatter(
        [2, 3, 4],
        paper_values,
        marker="D",
        s=72,
        color="#222222",
        label="paper Table 5",
        zorder=5,
    )
    uniform = macro[(3, "uniform_matrix", 20260715)]
    ax.scatter(
        [3],
        [uniform],
        marker="X",
        s=90,
        color="#e45756",
        label=f"3-Ran uniform ({uniform:.4f})",
        zorder=6,
    )
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("Prototype set size |S|")
    ax.set_ylabel("11-dataset macro accuracy")
    ax.set_ylim(0.70, 0.755)
    ax.set_title("Prototype-count ablation: robust peak, seed-sensitive exact values", loc="left")
    ax.legend(frameon=False, ncols=2, fontsize=8)
    _save(fig, images / "prototype-count-ablation.png")


def runtime_figure(artifacts: Path, images: Path) -> None:
    payload = _read(artifacts / "claim_5" / "raw_runtime_benchmark.json")
    ratios = [
        row
        for row in payload["benchmark"]["ratios"]
        if row["clock"] == "cpu_seconds"
    ]
    comparator_order = ["fgbcc", "ds", "bwa"]
    by_comparator = {row["comparator"]: row for row in ratios}
    medians = [float(by_comparator[name]["median"]) for name in comparator_order]
    lower = [
        float(by_comparator[name]["bootstrap_95_percent"][0])
        for name in comparator_order
    ]
    upper = [
        float(by_comparator[name]["bootstrap_95_percent"][1])
        for name in comparator_order
    ]
    x = np.arange(len(comparator_order))
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.errorbar(
        x,
        medians,
        yerr=[
            np.asarray(medians) - np.asarray(lower),
            np.asarray(upper) - np.asarray(medians),
        ],
        fmt="o",
        markersize=10,
        capsize=6,
        linewidth=2,
        color="#b279a2",
    )
    ax.axhline(
        0.10,
        color="#e45756",
        linestyle="--",
        linewidth=2,
        label="paper threshold (<0.10)",
    )
    for index, value in enumerate(medians):
        ax.text(index, value * 1.18, f"{value:.3f}×", ha="center", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.07, 12)
    ax.set_xticks(x, ["FGBCC", "released DS", "BWA"])
    ax.set_ylabel("PTBCC / comparator process time (log scale)")
    ax.set_title("Local CPU timing: every controlled ratio exceeds 10%", loc="left")
    ax.legend(frameon=False)
    ax.text(
        0.01,
        -0.19,
        "Five clean-process repetitions · rotated order · all 11 datasets · 10k bootstrap draws",
        transform=ax.transAxes,
        color="#555555",
    )
    _save(fig, images / "runtime-ratios.png")


def synthetic_figure(artifacts: Path, images: Path) -> None:
    payload = _read(artifacts / "claim_1" / "raw_synthetic_recovery.json")
    trials = payload["trials"]
    mv = np.asarray([float(row["mv_accuracy"]) for row in trials])
    ptbcc = np.asarray([float(row["ptbcc_accuracy"]) for row in trials])
    mae = np.asarray(
        [float(row["prototype_mae_after_matching"]) for row in trials]
    )
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    scatter = ax.scatter(
        mv,
        ptbcc,
        c=mae,
        cmap="viridis_r",
        s=52,
        edgecolor="white",
        linewidth=0.6,
    )
    lo = min(mv.min(), ptbcc.min()) - 0.005
    hi = max(mv.max(), ptbcc.max()) + 0.005
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#777777", label="equal accuracy")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Majority-vote accuracy")
    ax.set_ylabel("PTBCC accuracy")
    ax.set_title("Synthetic mechanism check: PTBCC wins in all 40 trials", loc="left")
    ax.legend(frameon=False)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Prototype MAE after Hungarian matching")
    ax.text(
        0.01,
        -0.19,
        f"Mean matched prototype MAE: {mae.mean():.6f}",
        transform=ax.transAxes,
        color="#555555",
    )
    _save(fig, images / "synthetic-recovery.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT / ".openresearch" / "artifacts",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "reports" / "ptbcc-claim-by-claim-2026-07-23",
    )
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    report_dir = args.report_dir.resolve()
    images = report_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    _style()
    val5_figure(artifacts, images)
    table4_figure(artifacts, images)
    ablation_figure(artifacts, images)
    runtime_figure(artifacts, images)
    synthetic_figure(artifacts, images)
    outputs = sorted(path.name for path in images.glob("*.png"))
    print(
        "REPORT_BUILD "
        + json.dumps(
            {
                "report_dir": str(report_dir.relative_to(ROOT)),
                "images": outputs,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
