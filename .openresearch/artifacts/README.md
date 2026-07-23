# PTBCC claim evidence

This directory is the durable evidence surface for the claim-by-claim
reproduction. Static contracts and methods are committed before execution.
Raw outputs are admitted only from completed `orx` run logs on a descendant
branch, then checked by the unchanged fixed command.

The fixed command inherited by every node is:

```text
uv sync --frozen && uv run python repro/src/run_ptbcc.py --output-dir outputs/full && uv run python -m unittest -v repro.tests.test_ptbcc
```

Python is pinned to 3.12 by `.python-version`; all packages are pinned by
`uv.lock`. Runs use local CPU unless an experiment description explicitly
records an authorized Hugging Face `cpu-upgrade` fallback.
