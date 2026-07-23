# PTBCC reproduction — KJq0iScNM6

CPU-only reproduction of *Let the Prototype Guide You: Robust Aggregation of
Sparse Multi-Class Annotations via Annotator Prototype Learning*.

The challenge catalog contains three claims. Two numeric claims are audited
against the primary paper source; the structural prototype claim is tested by
reconstructing equations (7)–(14), evaluating six exact cited public datasets,
and recovering known prototypes on synthetic data generated from the model.

Run:

```bash
uv sync --frozen
uv run python repro/src/run_ptbcc.py --output-dir outputs/full
uv run python -m unittest -v repro.tests.test_ptbcc
```
