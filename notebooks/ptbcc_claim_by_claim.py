import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # PTBCC: a claim-by-claim reproduction

    **Paper:** *Let the Prototype Guide You: Robust Aggregation of Sparse
    Multi-Class Annotations via Annotator Prototype Learning*

    This notebook opens with the strongest result already embedded. Nothing
    expensive needs to run: the small arrays below are copied from the
    immutable OpenResearch evidence, and the formal reproduction remains the
    fixed CPU command documented in the repository.
    """)
    return


@app.cell
def _(np, plt):
    val5_methods = ["MV", "BWA", "FGBCC", "DS", "PTBCC"]
    val5_accuracy = np.asarray([0.3516666667, 0.35, 0.38, 0.41, 0.56])
    val5_colors = ["#8c8c8c", "#4c78a8", "#54a24b", "#f58518", "#b279a2"]
    val5_fig, val5_ax = plt.subplots(figsize=(8.8, 4.2))
    val5_bars = val5_ax.bar(val5_methods, val5_accuracy, color=val5_colors)
    for val5_bar, val5_value in zip(val5_bars, val5_accuracy, strict=True):
        val5_ax.text(
            val5_bar.get_x() + val5_bar.get_width() / 2,
            val5_value + 0.012,
            f"{val5_value:.3f}",
            ha="center",
        )
    val5_ax.annotate(
        "+15.0 percentage points",
        xy=(4, 0.56),
        xytext=(3.0, 0.66),
        arrowprops={"arrowstyle": "->"},
        fontweight="bold",
    )
    val5_ax.set_ylim(0, 0.72)
    val5_ax.set_ylabel("Fractional-tie accuracy")
    val5_ax.set_title("Exact Val5: PTBCC versus faithful baselines", loc="left")
    val5_ax.spines[["top", "right"]].set_visible(False)
    val5_ax.grid(axis="y", alpha=0.2)
    val5_fig.tight_layout()
    val5_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Val5 contains exactly 100 tasks, 38 annotators, 5 classes, 1,000 labels,
    and 100 truths. Reconstructed PTBCC reaches **0.56**; the strongest
    faithfully released baseline is the survey's Dawid–Skene at **0.41**.
    That directly verifies the paper's largest reported gain.

    ## Why prototypes?

    A per-annotator confusion model estimates a `[W, K, K]` tensor. PTBCC
    instead learns a small shared `[S, K, K]` prototype tensor and a
    `[W, S]` mixture for annotators. Sparse annotators therefore borrow
    statistical strength from shared behavior patterns.
    """)
    return


@app.cell
def _(mo):
    claim_records = {
        "Claim 1 — architecture": {
            "verdict": "VERIFIED",
            "paper": "Annotators share prototype confusion matrices.",
            "observed": "Two normalized shared prototypes; normalized annotator mixtures; 40/40 synthetic wins; prototype MAE 0.0417523.",
            "why": "The implementation and independent file-only checker directly validate the claimed parameterization.",
        },
        "Claim 2 — Val5": {
            "verdict": "VERIFIED",
            "paper": "Up to 15 percentage points over the best baseline.",
            "observed": "PTBCC 0.56 versus released DS 0.41: +15.0 points.",
            "why": "Exact corpus and exact quantified gain.",
        },
        "Claim 3 — Table 4": {
            "verdict": "BLOCKED",
            "paper": "PTBCC .7472, FGBCC .7175, BWA .7132, MV .6986.",
            "observed": "PTBCC .7471525, FGBCC .7175817, BWA .7131503, MV .6986048.",
            "why": "Three match at four decimals; FGBCC rounds to .7176, and full author-side numerical provenance is absent.",
        },
        "Claim 4 — prototype count": {
            "verdict": "BLOCKED",
            "paper": "S=2 peaks at .7472; S=3 .7300; S=4 .7271.",
            "observed": "S=2 peaks in all three seeds; S=3 and S=4 vary materially with the unpublished seed.",
            "why": "The qualitative peak is robust, but the exact stochastic values cannot be attributed.",
        },
        "Claim 5 — efficiency": {
            "verdict": "BLOCKED",
            "paper": "About +3 points with under 10% of confusion-matrix runtime.",
            "observed": "+2.9571 points; local PTBCC/FGBCC CPU ratio 0.2258 [0.2007, 0.2294].",
            "why": "The local threshold is contradicted, but the paper omits its denominator/raw timings and used different hardware/original loops.",
        },
    }
    claim_picker = mo.ui.dropdown(
        options=list(claim_records),
        value="Claim 2 — Val5",
        label="Inspect a claim",
    )
    claim_picker
    return claim_picker, claim_records


@app.cell
def _(claim_picker, claim_records, mo):
    selected_claim = claim_records[claim_picker.value]
    verdict_color = (
        "#19753b" if selected_claim["verdict"] == "VERIFIED" else "#9c5700"
    )
    mo.md(
        f"""
        ### {claim_picker.value}

        <div style="border-left: 5px solid {verdict_color}; padding: 0.2rem 1rem;">
        <b>{selected_claim["verdict"]}</b><br><br>
        <b>Paper:</b> {selected_claim["paper"]}<br>
        <b>Observed:</b> {selected_claim["observed"]}<br>
        <b>Assessment:</b> {selected_claim["why"]}
        </div>
        """
    )
    return


@app.cell
def _(np, plt):
    paper_macro = np.asarray([0.6986, 0.7132, 0.7175, 0.7472])
    observed_macro = np.asarray(
        [0.6986047829, 0.7131502844, 0.7175816744, 0.7471525427]
    )
    macro_methods = ["MV", "BWA", "FGBCC", "PTBCC"]
    macro_x = np.arange(len(macro_methods))
    macro_fig, macro_ax = plt.subplots(figsize=(8.8, 4.3))
    macro_width = 0.34
    macro_ax.bar(
        macro_x - macro_width / 2,
        paper_macro,
        macro_width,
        label="paper",
        color="#d9d9d9",
        edgecolor="#777777",
    )
    macro_ax.bar(
        macro_x + macro_width / 2,
        observed_macro,
        macro_width,
        label="regenerated",
        color=["#8c8c8c", "#4c78a8", "#54a24b", "#b279a2"],
    )
    macro_ax.set_xticks(macro_x, macro_methods)
    macro_ax.set_ylim(0.68, 0.76)
    macro_ax.set_ylabel("11-dataset macro accuracy")
    macro_ax.set_title("Table 4: exact values versus regenerated values", loc="left")
    macro_ax.legend(frameon=False)
    macro_ax.spines[["top", "right"]].set_visible(False)
    macro_ax.grid(axis="y", alpha=0.2)
    macro_fig.tight_layout()
    macro_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How to reproduce the formal evidence

    ```bash
    uv sync --frozen
    uv run python repro/src/run_ptbcc.py --output-dir outputs/full
    uv run python -m unittest -v repro.tests.test_ptbcc
    ```

    The formal OpenResearch nodes use the same one-line command on local CPU.
    Every claim has a contract, raw JSON/CSV, source audit, independent
    checker output, negative-control output, provenance, and an exact
    `VERIFIED` or `BLOCKED` verdict under `.openresearch/artifacts/`.

    The full illustrated report is
    [`reports/ptbcc-claim-by-claim-2026-07-23/report.md`](../reports/ptbcc-claim-by-claim-2026-07-23/report.md).
    """)
    return


if __name__ == "__main__":
    app.run()
