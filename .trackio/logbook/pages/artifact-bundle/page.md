# Artifact bundle


---
<!-- trackio-cell
{"type": "dashboard", "id": "cell_f08f08ffe214", "created_at": "2026-07-16T15:09:41+00:00", "title": "Dashboard: ptbcc-repro", "dashboard_project": "ptbcc-repro"}
-->
**🎯 Trackio dashboard** `ptbcc-repro`

trackio-local-dashboard://ptbcc-repro


---
<!-- trackio-cell
{"type": "code", "id": "cell_89bf69d54865", "created_at": "2026-07-16T15:09:42+00:00", "title": "Run: python log_bundle.py (exit 0)", "command": ["python", "repro/src/log_bundle.py"], "exit_code": 0, "duration_s": 0.915}
-->
````bash
$ python repro/src/log_bundle.py
````

exit 0 · 0.9s


````python title=log_bundle.py
"""Register the compact PTBCC evidence bundle with Trackio."""

from pathlib import Path

import trackio


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    trackio.init(
        project="ptbcc-repro",
        name="cpu-full-evidence",
        config={"openreview_id": "KJq0iScNM6", "compute": "CPU"},
        auto_log_cpu=False,
    )
    artifact = trackio.Artifact(
        name="repro-bundle",
        type="dataset",
        description="Executed outputs, implementation, tests, and pinned paper source.",
        metadata={"openreview_id": "KJq0iScNM6", "arxiv_id": "2508.02123"},
    )
    artifact.add_dir(ROOT / "outputs" / "full", name="outputs/full")
    artifact.add_file(ROOT / "repro" / "src" / "run_ptbcc.py", name="repro/src/run_ptbcc.py")
    artifact.add_file(ROOT / "repro" / "tests" / "test_ptbcc.py", name="repro/tests/test_ptbcc.py")
    artifact.add_file(
        ROOT / "source" / "Formatting-Instructions-LaTeX-2026.tex",
        name="paper-source/main.tex",
    )
    trackio.log_artifact(artifact, aliases=["full"])
    trackio.finish()


if __name__ == "__main__":
    main()

````


````output
* Trackio project initialized: ptbcc-repro
* Trackio metrics logged to the local Trackio cache.
* View dashboard by running in your terminal:
[1m[38;5;208mtrackio show --project "ptbcc-repro"[0m
* or by running in Python: trackio.show(project="ptbcc-repro")
* Created new run: cpu-full-evidence
* Run finished. Uploading logs to Trackio (please wait...)

````
