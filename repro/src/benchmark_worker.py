"""Clean-process worker for the controlled all-dataset CPU benchmark."""

from __future__ import annotations

import argparse
import json
import time

from repro.src.reference_baselines import fit_bwa, fit_fgbcc, fit_published_ds
from repro.src.run_ptbcc import accuracy, fit_ptbcc, load_paper_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", required=True, choices=("bwa", "ds", "fgbcc", "ptbcc")
    )
    args = parser.parse_args()
    datasets = load_paper_datasets()
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        if args.method == "bwa":
            fit = fit_bwa(dataset)
            scores = fit.scores
        elif args.method == "ds":
            fit = fit_published_ds(dataset)
            scores = fit.scores
        elif args.method == "fgbcc":
            fit = fit_fgbcc(dataset)
            scores = fit.scores
        else:
            fit = fit_ptbcc(dataset)
            scores = fit.phi
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
        rows.append(
            {
                "dataset": dataset.name,
                "accuracy": accuracy(dataset, scores),
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
                "iterations": fit.iterations,
            }
        )
    print(
        "BENCHMARK_WORKER "
        + json.dumps({"method": args.method, "rows": rows}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
