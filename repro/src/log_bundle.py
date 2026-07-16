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
