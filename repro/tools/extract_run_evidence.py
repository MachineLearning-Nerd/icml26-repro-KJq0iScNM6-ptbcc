#!/usr/bin/env python3
"""Materialize claim-local evidence from one immutable OpenResearch run log.

The OpenResearch local-mode log is the authoritative result channel.  This
utility copies only named JSON markers into reviewable files and records a hash
of the complete log response so the extraction remains auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
FIXED_COMMAND = (
    "uv sync --frozen && uv run python repro/src/run_ptbcc.py "
    "--output-dir outputs/full && uv run python -m unittest -v "
    "repro.tests.test_ptbcc"
)


def _run(*args: str) -> str:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{' '.join(args)} failed ({completed.returncode}): "
            f"{completed.stderr[-2000:]}"
        )
    return completed.stdout


def _markers(log: str) -> dict[str, list[Any]]:
    wanted = {
        "DATASET_RESULT",
        "FGBCC_REFERENCE_VALIDATION",
        "ABLATION_RAW",
        "SYNTHETIC_RAW",
        "ABLATION_MACRO",
        "RUNTIME_BENCHMARK",
        "CLAIM_RESULTS",
        "CLAIM_VERIFIER_DETAILS",
    }
    parsed: dict[str, list[Any]] = {marker: [] for marker in wanted}
    for line in log.splitlines():
        marker, separator, payload = line.partition(" ")
        if separator and marker in wanted:
            parsed[marker].append(json.loads(payload))
    counts = {key: len(value) for key, value in parsed.items()}
    expected = {
        "DATASET_RESULT": 11,
        "FGBCC_REFERENCE_VALIDATION": 1,
        "ABLATION_RAW": 1,
        "SYNTHETIC_RAW": 1,
        "ABLATION_MACRO": 1,
        "RUNTIME_BENCHMARK": 1,
        "CLAIM_RESULTS": 1,
    }
    required_counts = {key: counts[key] for key in expected}
    if required_counts != expected:
        raise ValueError(f"unexpected marker counts: {counts}, expected {expected}")
    if counts["CLAIM_VERIFIER_DETAILS"] not in (0, 1):
        raise ValueError(
            "expected at most one CLAIM_VERIFIER_DETAILS marker, observed "
            f"{counts['CLAIM_VERIFIER_DETAILS']}"
        )
    return parsed


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    log = _run("orx", "logs", args.run_id, "--bytes", "200000")
    status = _run("orx", "exp", "status", args.experiment_id)
    if args.run_id not in status or args.commit not in status:
        raise ValueError("run id or full commit is absent from experiment status")
    if f"command:  {FIXED_COMMAND}" not in status:
        raise ValueError("experiment does not use the frozen command")
    parsed = _markers(log)
    rows = parsed["DATASET_RESULT"]
    val5 = next(row for row in rows if row["dataset"] == "Val5")
    claims = parsed["CLAIM_RESULTS"][0]

    provenance = {
        "project_id": "6c85b7b1-64e2-437d-8936-2229c4cffb13",
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "git_commit": args.commit,
        "backend": "local",
        "compute": "CPU only",
        "fixed_command": FIXED_COMMAND,
        "log_response_sha256": hashlib.sha256(log.encode()).hexdigest(),
        "experiment_status": status.strip(),
        "environment_inputs": {
            ".python-version": (ROOT / ".python-version").read_text().strip(),
            "pyproject_sha256": hashlib.sha256(
                (ROOT / "pyproject.toml").read_bytes()
            ).hexdigest(),
            "uv_lock_sha256": hashlib.sha256(
                (ROOT / "uv.lock").read_bytes()
            ).hexdigest(),
        },
        "deterministic_seeds": {
            "ptbcc_main": 20260715,
            "ablation": [20260715, 20260716, 20260717],
            "synthetic": list(range(73000, 73040)),
            "bootstrap": 20260723,
        },
    }
    for claim_id in range(1, 6):
        _write_json(
            ARTIFACTS / f"claim_{claim_id}" / "run_provenance.json",
            provenance,
        )

    _write_json(
        ARTIFACTS / "claim_1" / "raw_synthetic_recovery.json",
        {
            "trials": parsed["SYNTHETIC_RAW"][0],
            "architecture_outputs": {
                "learned_prototype_tensor": "(n_prototypes, n_classes, n_classes)",
                "learned_worker_mixture_tensor": "(n_workers, n_prototypes)",
            },
        },
    )
    _write_json(
        ARTIFACTS / "claim_2" / "raw_val5_result.json",
        {
            "dataset_result": val5,
            "claim_result": claims["claim_2"],
        },
    )
    _write_json(
        ARTIFACTS / "claim_3" / "raw_table4_results.json",
        {
            "dataset_results": rows,
            "claim_result": claims["claim_3"],
            "fgbcc_reference_validation": parsed[
                "FGBCC_REFERENCE_VALIDATION"
            ][0],
        },
    )
    _write_json(
        ARTIFACTS / "claim_4" / "raw_ablation.json",
        {
            "raw": parsed["ABLATION_RAW"][0],
            "macro": parsed["ABLATION_MACRO"][0],
            "claim_result": claims["claim_4"],
        },
    )
    _write_json(
        ARTIFACTS / "claim_5" / "raw_runtime_benchmark.json",
        {
            "benchmark": parsed["RUNTIME_BENCHMARK"][0],
            "claim_result": claims["claim_5_controlled"],
        },
    )
    verifier_details = parsed["CLAIM_VERIFIER_DETAILS"]
    if verifier_details:
        details = verifier_details[0]
        for claim_id in range(1, 6):
            claim_key = str(claim_id)
            evaluation = details["evaluations"][claim_key]
            control = details["negative_controls"][claim_key]
            _write_json(
                ARTIFACTS
                / f"claim_{claim_id}"
                / "independent_checker.json",
                evaluation,
            )
            _write_json(
                ARTIFACTS / f"claim_{claim_id}" / "verifier_output.json",
                evaluation,
            )
            _write_json(
                ARTIFACTS / f"claim_{claim_id}" / "negative_control.json",
                control,
            )
        provenance["checker_runtime"] = details["provenance"]
        for claim_id in range(1, 6):
            _write_json(
                ARTIFACTS / f"claim_{claim_id}" / "run_provenance.json",
                provenance,
            )
    print(
        json.dumps(
            {
                "status": "ok",
                "run_id": args.run_id,
                "commit": args.commit,
                "log_response_sha256": provenance["log_response_sha256"],
                "files_written": 25 if verifier_details else 10,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
