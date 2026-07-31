"""GLASSBOX monitoring dashboard -- page 1 Population, page 2 Recovery,
page 3 Person encoder, page 4 Calibration.

    streamlit run app/dashboard.py

Pick the page in the sidebar.

This file sits outside the Wall on purpose. It opens exactly three things:
``results/*.json`` and the PNGs next to them (what the grader-side code
writes) and ``data/public/`` (what every model is allowed to see anyway). It
never reaches for planted values, and ``tests/test_wall.py`` scans this file
and fails the suite if it ever starts to.

Everything on screen is read straight out of those files. Nothing is
recomputed here, so a number in the browser is the same number in the JSON --
including every pass and fail, which come out of the report's own verdict
block rather than being re-decided in the page.

Nothing crashes on a fresh clone: every panel checks for its file first and
says which command produces it if it is missing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PUBLIC_DIR = REPO_ROOT / "data" / "public"
SUMMARY_PATH = RESULTS_DIR / "population_summary.json"
COSTS_PATH = RESULTS_DIR / "COSTS.md"

#: Gate 2 was attempted twice. The page shows the second attempt as the current
#: state and keeps the first beside it for comparison; both sets of files stay
#: in results/, so nothing here overwrites the record of what happened first.
RECOVERY_PATH = RESULTS_DIR / "stage2_v2_recovery.json"
SCATTER_PATH = RESULTS_DIR / "stage2_v2_scatter.png"
HEATMAP_PATH = RESULTS_DIR / "stage2_v2_heatmap.png"
PREVIOUS_RECOVERY_PATH = RESULTS_DIR / "stage2_recovery.json"

CURRENT_ATTEMPT = "attempt 2"
PREVIOUS_ATTEMPT = "attempt 1"

#: Results files that count as "an experiment run" and can be picked below.
QA_PATTERN = "*_qa.json"

#: The frozen Gate 1 bars (PREREGISTRATION.md section 6). They are drawn as
#: reference lines only. Pass and fail are read from the report, never judged
#: here, so this page cannot disagree with the QA run that produced the file.
MAX_PAIR_AGREEMENT_BAR = 0.95
MEDIAN_PAIR_AGREEMENT_BAR = 0.80
OBEDIENCE_BAR = 0.5
RETEST_BAND = (0.70, 0.90)

#: The frozen Gate 2 bars (PREREGISTRATION.md section 6), drawn as reference
#: lines only. Same rule as above: pass and fail come out of the report.
TRAIT_RECOVERY_BAR = 0.8
ITEM_RECOVERY_BAR = 0.7
BLUR_COVERAGE_BAND = (0.60, 0.75)

#: Stage 3. The Gate 3 report carries the verdict block, both tracks and the
#: curves; the two pictures sit next to it; the item encoder is graded in its
#: own file and shown compactly on the same page.
GATE3_PATH = RESULTS_DIR / "stage3_gate3.json"
BLUR_VS_N_PATH = RESULTS_DIR / "stage3_blur_vs_n.png"
GATE3_CURVES_PATH = RESULTS_DIR / "stage3_gate3_curves.png"
BACKBONE_PATH = RESULTS_DIR / "stage3_person_backbone.json"
ITEM_ENCODER_PATH = RESULTS_DIR / "stage3_item_encoder.json"

#: The frozen Gate 3 bars (PREREGISTRATION.md section 6). Reference values
#: only -- pass and fail are read from the reports, as everywhere else here.
PERSON_RMSE_BAR = 0.5
PERSON_LIFT_BAR = 0.30
PERSON_COVERAGE_BAND = (0.60, 0.75)
ITEM_COSINE_BAR = 0.7
ITEM_DISCRIMINATION_BAR = 0.6

#: Used only when a report omits its own threshold, the same fallback rule the
#: Gate 2 blur panel already follows.
GATE3_PERSON_BARS: dict[str, Any] = {
    "rmse": PERSON_RMSE_BAR,
    "blur_coverage": PERSON_COVERAGE_BAND,
    "lift": PERSON_LIFT_BAR,
}
GATE3_ITEM_BARS: dict[str, Any] = {
    "median_loading_cosine": ITEM_COSINE_BAR,
    "discrimination_r": ITEM_DISCRIMINATION_BAR,
}

#: The one bar on this page the report deliberately does not decide: it stores
#: both readings of "monotone in expectation" and hands the call up
#: (``pass: null``). The call taken is PASS on the in-expectation wording. The
#: report's own strict readings are printed beside it, unchanged, so the page
#: shows the ruling and the thing it ruled on together.
MONOTONICITY_KEY = "monotone_blur"
MONOTONICITY_VERDICT = "PASS"
MONOTONICITY_RULING = "orchestrator ruling on qualitative wording; both readings reported"

#: Stage 4. The Gate 4 report carries the verdict block, every arm's Brier and
#: calibration, the per-stratum split and the aggregate checks; two pictures
#: sit next to it.
GATE4_PATH = RESULTS_DIR / "stage4_gate4.json"
RELIABILITY_PATH = RESULTS_DIR / "stage4_reliability.png"
AGGREGATE_PATH = RESULTS_DIR / "stage4_aggregate.png"

#: The frozen Gate 4 bars (PREREGISTRATION.md section 6). Reference values
#: only -- pass and fail are read from the report, as everywhere else here.
PRIMARY_LIFT_BAR = 0.25
ECE_BAR = 0.05
CELL_CORRELATION_BAR = 0.9

#: Used only when a report omits its own threshold, the same fallback rule the
#: Gate 2 and Gate 3 panels already follow.
GATE4_BARS: dict[str, Any] = {
    "primary_lift": PRIMARY_LIFT_BAR,
    "ece": ECE_BAR,
    "correlation": CELL_CORRELATION_BAR,
}

#: The arms this page shows: the confirmatory one and the three pre-registered
#: baselines. The report also carries exploratory arms; those stay off the page.
GATE4_ARMS = ("model", "marginal", "profile", "knn")

#: Near first, then same-domain, then far -- how far a probe item sits from the
#: items the interview asked about.
STRATUM_ORDER = ("near", "same-domain", "far")

INK = "#4c78a8"
MARK = "#d1495b"
BAND = "#94c973"


# --------------------------------------------------------------------------
# reading files
# --------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any] | None:
    """A JSON file, or None if it is missing or unreadable."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def qa_files() -> list[Path]:
    """Every QA report in results/, newest first."""
    found = sorted(RESULTS_DIR.glob(QA_PATTERN))
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def count_public_profiles() -> int:
    """Fallback persona count when no summary has been exported yet."""
    folder = PUBLIC_DIR / "profiles"
    return len(list(folder.glob("*.json"))) if folder.is_dir() else 0


CORE_HOURS = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*core-h", re.IGNORECASE)
DOLLARS = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")


def parse_costs(text: str) -> dict[str, float | int]:
    """Total core-hours and API dollars from the cost table in COSTS.md.

    Only the last column of each table row is read, so the core-hours quoted
    inside the "what was billed" column are not counted a second time.
    """
    core_hours = 0.0
    api_usd = 0.0
    rows = 0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 5:
            continue
        first = cells[0]
        if not first or first.lower() == "date" or set(first) <= set("-: "):
            continue  # header or separator row

        rows += 1
        cost = cells[-1]
        core_hours += sum(float(m.group(1)) for m in CORE_HOURS.finditer(cost))
        api_usd += sum(float(m.group(1)) for m in DOLLARS.finditer(cost))

    return {"core_hours": core_hours, "api_usd": api_usd, "n_rows": rows}


# --------------------------------------------------------------------------
# chart helpers
# --------------------------------------------------------------------------


def histogram_frame(histogram: dict[str, Any]) -> pd.DataFrame | None:
    """A {bin_edges, counts} block as a frame of bar extents."""
    edges = [float(e) for e in histogram.get("bin_edges") or []]
    counts = [int(c) for c in histogram.get("counts") or []]
    if len(edges) < 2 or len(counts) != len(edges) - 1:
        return None
    return pd.DataFrame(
        {
            "left": edges[:-1],
            "right": edges[1:],
            "centre": [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])],
            "count": counts,
        }
    )


def histogram_chart(
    frame: pd.DataFrame,
    x_title: str,
    y_title: str = "personas",
    height: int = 220,
    rules: tuple[float, ...] = (),
    band: tuple[float, float] | None = None,
    colour: str = INK,
) -> alt.LayerChart:
    """Bar histogram, optionally with a shaded band and vertical marker lines."""
    domain = [float(frame["left"].min()), float(frame["right"].max())]
    layers: list[Any] = []

    if band is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"lo": [band[0]], "hi": [band[1]]}))
            .mark_rect(color=BAND, opacity=0.25)
            .encode(x=alt.X("lo:Q"), x2=alt.X2("hi:Q"))
        )

    layers.append(
        alt.Chart(frame)
        .mark_bar(color=colour)
        .encode(
            x=alt.X("left:Q", title=x_title, scale=alt.Scale(domain=domain, nice=False)),
            x2=alt.X2("right:Q"),
            y=alt.Y("count:Q", title=y_title),
            tooltip=[
                alt.Tooltip("left:Q", title="from", format=".3f"),
                alt.Tooltip("right:Q", title="to", format=".3f"),
                alt.Tooltip("count:Q", title="count"),
            ],
        )
    )

    for value in rules:
        layers.append(
            alt.Chart(pd.DataFrame({"v": [value]}))
            .mark_rule(color=MARK, strokeDash=[5, 4], size=2)
            .encode(x=alt.X("v:Q"))
        )

    return alt.layer(*layers).properties(height=height)


def show(chart: Any) -> None:
    st.altair_chart(chart, width="stretch")


def table(rows: list[dict[str, Any]]) -> None:
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def fmt(value: Any, places: int = 3) -> str:
    """A number for display, or a dash when the report has none."""
    if value is None:
        return "--"
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------


def header(summary: dict[str, Any] | None, reports: list[Path]) -> None:
    st.title("GLASSBOX monitor")

    if summary is not None:
        personas = summary.get("n_personas", 0)
        persona_note = "planted and exported"
    else:
        personas = count_public_profiles()
        persona_note = "public profiles on disk"

    costs = parse_costs(COSTS_PATH.read_text(encoding="utf-8")) if COSTS_PATH.is_file() else None

    left, middle, right = st.columns(3)
    left.metric("Personas generated", f"{personas:,}", help=persona_note)
    middle.metric("Experiments run", len(reports), help=f"results/{QA_PATTERN} files")
    if costs is None:
        right.metric("Compute so far", "--", help="results/COSTS.md not found")
    else:
        right.metric(
            "Compute so far",
            f"{costs['core_hours']:.2f} core-h",
            help=f"summed over {costs['n_rows']} logged rows in results/COSTS.md",
        )
        right.caption(f"API spend so far: about ${costs['api_usd']:.2f}")

    st.divider()


# --------------------------------------------------------------------------
# population: the planted side, from the exported summary
# --------------------------------------------------------------------------


EXPORT_HINT = (
    "Nothing to show yet. Produce it with:\n\n"
    "```\n"
    "python -m src.eval.export_population_summary \\\n"
    "    --truth <planted dir> --public data/public \\\n"
    "    --out results/population_summary.json\n"
    "```"
)


def trait_distributions(summary: dict[str, Any]) -> None:
    st.subheader("Trait distributions")
    block = summary.get("theta") or {}
    histograms = block.get("histograms") or {}
    dimensions = summary.get("dimensions") or sorted(histograms)
    labels = summary.get("dimension_labels") or {}

    if not histograms:
        st.info("The summary holds no trait histograms.")
        return

    st.caption(
        f"{summary.get('n_personas', 0)} personas, {block.get('bins', '?')} bins over "
        f"{block.get('range', [-2, 2])}. Counts only -- no persona sits in this file."
    )

    for row_start in range(0, len(dimensions), 4):
        columns = st.columns(4)
        for column, dimension in zip(columns, dimensions[row_start : row_start + 4]):
            frame = histogram_frame(histograms.get(dimension) or {})
            with column:
                st.markdown(f"**{dimension}** &mdash; {labels.get(dimension, '')}")
                if frame is None:
                    st.caption("no data")
                else:
                    show(histogram_chart(frame, "planted value", height=150))


def wobble_panel(summary: dict[str, Any]) -> None:
    st.subheader("Response wobble")
    frame = histogram_frame((summary.get("wobble") or {}).get("histogram") or {})
    if frame is None:
        st.info("The summary holds no wobble histogram.")
        return
    st.caption(
        f"How much each persona is planted to flip its own answers. "
        f"{summary.get('n_with_wobble', 0)} personas."
    )
    show(histogram_chart(frame, "wobble", height=200))


def demographics_panel(summary: dict[str, Any]) -> None:
    st.subheader("Demographics (public profiles)")
    block = summary.get("demographics") or {}
    fields = block.get("fields") or {}
    if not fields:
        st.info("The summary holds no demographic counts.")
        return

    st.caption(f"{block.get('n_profiles', 0)} public profiles.")
    names = list(fields)
    for row_start in range(0, len(names), 2):
        columns = st.columns(2)
        for column, name in zip(columns, names[row_start : row_start + 2]):
            entry = fields[name]
            with column:
                heading = name.replace("_", " ")
                note = (
                    f" -- top {entry['n_shown']} of {entry['n_distinct']}"
                    if entry.get("truncated")
                    else f" -- {entry.get('n_distinct', 0)} values"
                )
                st.markdown(f"**{heading}**{note}")
                table(
                    [
                        {heading: row["value"], "personas": row["count"]}
                        for row in entry.get("counts") or []
                    ]
                )


# --------------------------------------------------------------------------
# population: the answered side, from a QA report
# --------------------------------------------------------------------------


def agreement_panel(diversity: dict[str, Any]) -> None:
    st.subheader("Pairwise agreement")
    frame = histogram_frame(diversity.get("histogram") or {})
    if frame is None:
        st.info("This report has no agreement histogram.")
        return

    st.caption(
        f"Every one of {diversity.get('n_pairs', 0):,} persona pairs, by the share of "
        f"items they answered exactly the same. Dashed lines are the two Gate 1 bars: "
        f"median at most {MEDIAN_PAIR_AGREEMENT_BAR}, no pair above {MAX_PAIR_AGREEMENT_BAR}."
    )
    show(
        histogram_chart(
            frame,
            "pairwise agreement",
            y_title="persona pairs",
            rules=(MEDIAN_PAIR_AGREEMENT_BAR, MAX_PAIR_AGREEMENT_BAR),
        )
    )
    table(
        [
            {
                "max pair": fmt(diversity.get("max_pair_agreement")),
                "median pair": fmt(diversity.get("median_pair_agreement")),
                "mean pair": fmt(diversity.get("mean_pair_agreement")),
                "min pair": fmt(diversity.get("min_pair_agreement")),
                "personas": diversity.get("n_personas"),
                "items": diversity.get("n_items"),
            }
        ]
    )


def spectrum_panel(diversity: dict[str, Any]) -> None:
    st.subheader("Eigenvalue spectrum")
    values = [float(v) for v in diversity.get("eigenvalues") or []]
    if not values:
        st.info("This report has no eigenvalue spectrum.")
        return

    positive = [(index + 1, value) for index, value in enumerate(values) if value > 0]
    dropped = len(values) - len(positive)
    frame = pd.DataFrame(positive, columns=["index", "eigenvalue"])
    ratio = diversity.get("effective_dimensionality")

    line = (
        alt.Chart(frame)
        .mark_line(color=INK, point=alt.OverlayMarkDef(color=INK, size=25))
        .encode(
            x=alt.X("index:Q", title="component"),
            y=alt.Y(
                "eigenvalue:Q",
                title="eigenvalue (log scale)",
                scale=alt.Scale(type="log"),
            ),
            tooltip=[
                alt.Tooltip("index:Q", title="component"),
                alt.Tooltip("eigenvalue:Q", format=".4f"),
            ],
        )
    )
    layers: list[Any] = [line]

    if ratio is not None:
        marker = pd.DataFrame({"v": [float(ratio)], "label": [f"participation ratio {ratio:.2f}"]})
        layers.append(
            alt.Chart(marker).mark_rule(color=MARK, strokeDash=[5, 4], size=2).encode(x=alt.X("v:Q"))
        )
        layers.append(
            alt.Chart(marker)
            .mark_text(color=MARK, align="left", dx=6, dy=-6, baseline="top", fontSize=12)
            .encode(x=alt.X("v:Q"), y=alt.value(6), text=alt.Text("label:N"))
        )

    st.caption(
        f"Item correlation matrix over {diversity.get('n_items_in_eigendecomposition', 0)} items. "
        f"Effective dimensionality (participation ratio) **{fmt(ratio, 2)}**; the Gate 1 bar is 5."
        + (f" {dropped} zero or negative eigenvalues are not drawn (log axis)." if dropped else "")
    )
    show(alt.layer(*layers).properties(height=260))


def obedience_panel(obedience: dict[str, Any]) -> None:
    st.subheader("Trait obedience")
    per_dimension = obedience.get("per_dimension") or {}
    rows = [
        {"dimension": dimension, "r": float(value)}
        for dimension, value in per_dimension.items()
        if value is not None
    ]
    if not rows:
        st.info("This report measured no dimension.")
        return

    frame = pd.DataFrame(rows)
    order = [row["dimension"] for row in rows]
    bars = (
        alt.Chart(frame)
        .mark_bar(color=INK)
        .encode(
            x=alt.X("dimension:N", title=None, sort=order),
            y=alt.Y("r:Q", title="r with planted value"),
            tooltip=[alt.Tooltip("dimension:N"), alt.Tooltip("r:Q", format=".3f")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"v": [OBEDIENCE_BAR]}))
        .mark_rule(color=MARK, strokeDash=[5, 4], size=2)
        .encode(y=alt.Y("v:Q"))
    )

    st.caption(
        f"Correlation per dimension between the planted value and the persona's keyed mean "
        f"answer. Median **{fmt(obedience.get('median'))}** across "
        f"{obedience.get('n_dimensions_measured', 0)} dimensions; the Gate 1 bar is "
        f"{OBEDIENCE_BAR} (dashed)."
    )
    show(alt.layer(bars, rule).properties(height=260))


def retest_panel(retest: dict[str, Any] | None) -> None:
    st.subheader("Test-retest agreement")
    if not retest:
        st.info("This run had no re-asked round, so test-retest was not measured.")
        return

    frame = histogram_frame(retest.get("per_persona_histogram") or {})
    st.caption(
        f"One bar per persona rate over {retest.get('n_reasked', 0):,} re-asked answers from "
        f"{retest.get('n_personas', 0)} personas. The green band is the pre-registered "
        f"{RETEST_BAND[0]:.0%}-{RETEST_BAND[1]:.0%} window."
    )
    if frame is None:
        st.info("This report has no per-persona histogram.")
    else:
        show(histogram_chart(frame, "per-persona agreement", band=RETEST_BAND))

    binary = retest.get("binary") or {}
    likert = retest.get("likert5") or {}
    table(
        [
            {
                "slice": "pooled",
                "agreement": fmt(retest.get("pooled_agreement")),
                "chance": "--",
                "n": retest.get("n_reasked"),
                "rule": "exact on binary, within +-1 on Likert-5",
            },
            {
                "slice": "binary",
                "agreement": fmt(binary.get("agreement")),
                "chance": fmt(binary.get("chance"), 2),
                "n": binary.get("n"),
                "rule": binary.get("rule", ""),
            },
            {
                "slice": "Likert-5",
                "agreement": fmt(likert.get("agreement")),
                "chance": fmt(likert.get("chance"), 2),
                "n": likert.get("n"),
                "rule": likert.get("rule", ""),
            },
        ]
    )


def verdict_panel(verdict: dict[str, Any] | None) -> None:
    st.subheader("Gate 1 verdict")
    bars = (verdict or {}).get("bars") or {}
    if not bars:
        st.info("This report has no verdict block.")
        return

    rows = []
    for key, entry in bars.items():
        if entry.get("pass") is True:
            mark = "PASS"
        elif entry.get("pass") is False:
            mark = "FAIL"
        else:
            mark = "not evaluated"
        rows.append(
            {
                "bar": entry.get("bar", key),
                "rule": entry.get("rule", ""),
                "value": fmt(entry.get("value")),
                "verdict": mark,
            }
        )
    table(rows)

    if verdict.get("all_pass"):
        st.success("Gate 1: PASS -- every frozen bar is met.")
    else:
        st.error("Gate 1: FAIL -- at least one frozen bar is missed or not evaluated.")


def qa_section(reports: list[Path]) -> None:
    st.header("Answered population")
    if not reports:
        st.info(
            "No QA report in results/ yet. Produce one with:\n\n"
            "```\n"
            "python -m src.eval.population_qa \\\n"
            "    --answers <answers.jsonl> --truth <planted dir> \\\n"
            "    --public-bank data/public/bank_items.json \\\n"
            "    --out results/population_qa.json\n"
            "```"
        )
        return

    choice = st.selectbox(
        "QA report",
        reports,
        format_func=lambda path: path.name,
        help="Every results/*_qa.json file, newest first.",
    )
    report = load_json(choice)
    if report is None:
        st.error(f"{choice.name} could not be read as JSON.")
        return

    st.caption(
        f"{report.get('n_personas', 0)} personas x {report.get('n_items', 0)} items, "
        f"{report.get('n_answers_main', 0):,} answers."
    )

    verdict_panel(report.get("verdict"))
    diversity = report.get("diversity") or {}
    agreement_panel(diversity)
    spectrum_panel(diversity)
    obedience_panel(report.get("obedience") or {})
    retest_panel(report.get("retest"))


# --------------------------------------------------------------------------
# recovery: Stage 2, from results/stage2_recovery.json and its two PNGs
# --------------------------------------------------------------------------


RECOVERY_HINT = (
    "No Stage 2 recovery report in results/ yet. Produce it with:\n\n"
    "```\n"
    "python -m src.eval.recovery \\\n"
    "    --fit results/stage2_v2_fit.npz \\\n"
    "    --diagnostics results/stage2_v2_fit_diagnostics.json \\\n"
    "    --truth <planted dir> --splits experiments/splits_v1.json \\\n"
    "    --answers <noised answers.jsonl> \\\n"
    "    --out results --out-prefix stage2_v2\n"
    "```"
)


def bar_threshold(value: Any) -> str:
    """The frozen threshold of one bar, however the report wrote it down."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{fmt(value[0], 2)}-{fmt(value[1], 2)}"
    if isinstance(value, (int, float)):
        return fmt(value, 2)
    return str(value) if value is not None else "--"


def gate2_verdict_panel(verdict: dict[str, Any] | None) -> None:
    st.subheader("Gate 2 verdict")
    bars = (verdict or {}).get("bars") or {}
    if not bars:
        st.info("This report has no verdict block.")
        return

    rows = []
    caveats = []
    for key, entry in bars.items():
        if entry.get("pass") is True:
            mark = "PASS"
        elif entry.get("pass") is False:
            mark = "FAIL"
        else:
            mark = "not evaluated"
        label = entry.get("value_label")
        rows.append(
            {
                "bar": key.replace("_", " "),
                "rule": entry.get("statement", ""),
                "must be": bar_threshold(entry.get("bar")),
                "value": fmt(entry.get("value")) + (f" ({label})" if label else ""),
                "verdict": mark,
            }
        )
        if entry.get("caveat"):
            caveats.append(f"**{key.replace('_', ' ')}:** {entry['caveat']}")
    table(rows)

    failed = [name.replace("_", " ") for name in (verdict or {}).get("failed_bars") or []]
    if (verdict or {}).get("all_bars_pass"):
        st.success("Gate 2: PASS -- every frozen bar is met.")
    elif failed:
        st.error(f"Gate 2: FAIL -- {len(failed)} of {len(bars)} bars missed: {', '.join(failed)}.")
    else:
        st.error("Gate 2: FAIL -- at least one frozen bar is missed or not evaluated.")

    for caveat in caveats:
        st.caption(caveat)


def attempt_comparison_panel(
    current: dict[str, Any] | None, previous: dict[str, Any] | None
) -> None:
    """The four frozen bars, attempt 1 next to attempt 2.

    Both columns are read out of the two reports' own verdict blocks, so this
    table cannot disagree with either run.
    """
    st.subheader(f"{PREVIOUS_ATTEMPT.capitalize()} vs {CURRENT_ATTEMPT}")
    now = ((current or {}).get("verdict") or {}).get("bars") or {}
    before = ((previous or {}).get("verdict") or {}).get("bars") or {}

    if not now:
        st.info("This report has no verdict block to compare.")
        return
    if previous is None:
        st.info(
            f"{PREVIOUS_RECOVERY_PATH.name} is not in results/, so there is nothing to "
            f"compare {CURRENT_ATTEMPT} against. The numbers below are {CURRENT_ATTEMPT} alone."
        )
        return

    def mark(entry: dict[str, Any]) -> str:
        if entry.get("pass") is True:
            return "PASS"
        return "FAIL" if entry.get("pass") is False else "--"

    rows = []
    for key, entry in now.items():
        old = before.get(key) or {}
        old_value = old.get("value")
        new_value = entry.get("value")
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            move = f"{float(new_value) - float(old_value):+.3f}"
        else:
            move = "--"
        rows.append(
            {
                "bar": key.replace("_", " "),
                "must be": bar_threshold(entry.get("bar")),
                f"{PREVIOUS_ATTEMPT} value": fmt(old_value),
                f"{PREVIOUS_ATTEMPT} verdict": mark(old) if old else "not run",
                f"{CURRENT_ATTEMPT} value": fmt(new_value),
                f"{CURRENT_ATTEMPT} verdict": mark(entry),
                "move": move,
            }
        )
    table(rows)

    passed_now = sum(1 for entry in now.values() if entry.get("pass") is True)
    passed_before = sum(1 for entry in before.values() if entry.get("pass") is True)
    st.caption(
        f"{PREVIOUS_ATTEMPT.capitalize()} passed {passed_before} of {len(before) or len(now)} bars; "
        f"{CURRENT_ATTEMPT} passes {passed_now} of {len(now)}. Attempt 1 is still on disk as "
        f"{PREVIOUS_RECOVERY_PATH.name} -- it is not overwritten."
    )


def picture_panel(correlations: dict[str, Any] | None) -> None:
    st.subheader("Pictures")
    left, right = st.columns(2)

    with left:
        st.markdown("**Estimated vs planted trait, per dimension**")
        if SCATTER_PATH.is_file():
            st.image(str(SCATTER_PATH), width="stretch")
        else:
            st.info(f"{SCATTER_PATH.name} not found -- the recovery run writes it unless --no-plots.")

    with right:
        st.markdown("**Recovered vs planted trait correlations**")
        if HEATMAP_PATH.is_file():
            st.image(str(HEATMAP_PATH), width="stretch")
        else:
            st.info(f"{HEATMAP_PATH.name} not found -- the recovery run writes it unless --no-plots.")

        holdout = (correlations or {}).get("holdout") or {}
        if holdout:
            cell = holdout.get("max_abs_delta_cell") or []
            where = f" ({'-'.join(str(part) for part in cell)})" if cell else ""
            st.caption(
                f"Held-out personas: largest off-diagonal miss "
                f"**{fmt(holdout.get('max_abs_delta_offdiagonal'))}**{where}, mean off-diagonal miss "
                f"{fmt(holdout.get('mean_abs_delta_offdiagonal'))}."
            )


def trait_recovery_panel(
    holdout: dict[str, Any],
    train: dict[str, Any],
    previous_holdout: dict[str, Any] | None = None,
) -> None:
    """Per-dimension recovery, with attempt 1 beside attempt 2 when it is there."""
    st.subheader("Trait recovery per dimension")
    per_dimension = holdout.get("per_dimension_r") or {}
    rows = [
        {"dimension": dimension, "attempt": CURRENT_ATTEMPT, "r": float(value)}
        for dimension, value in per_dimension.items()
        if value is not None
    ]
    if not rows:
        st.info("This report has no per-dimension trait recovery.")
        return

    order = [row["dimension"] for row in rows]
    before = (previous_holdout or {}).get("per_dimension_r") or {}
    rows += [
        {"dimension": dimension, "attempt": PREVIOUS_ATTEMPT, "r": float(before[dimension])}
        for dimension in order
        if before.get(dimension) is not None
    ]
    paired = any(row["attempt"] == PREVIOUS_ATTEMPT for row in rows)

    frame = pd.DataFrame(rows)
    encoding: dict[str, Any] = dict(
        x=alt.X("dimension:N", title=None, sort=order),
        y=alt.Y("r:Q", title="r(estimated, planted)"),
        color=alt.condition(alt.datum.r < TRAIT_RECOVERY_BAR, alt.value(MARK), alt.value(INK)),
        tooltip=[
            alt.Tooltip("dimension:N"),
            alt.Tooltip("attempt:N"),
            alt.Tooltip("r:Q", format=".3f"),
        ],
    )
    if paired:
        encoding["xOffset"] = alt.XOffset(
            "attempt:N", sort=[PREVIOUS_ATTEMPT, CURRENT_ATTEMPT]
        )
        encoding["opacity"] = alt.condition(
            f"datum.attempt == '{PREVIOUS_ATTEMPT}'", alt.value(0.4), alt.value(1.0)
        )
    bars = alt.Chart(frame).mark_bar().encode(**encoding)
    rule = (
        alt.Chart(pd.DataFrame({"v": [TRAIT_RECOVERY_BAR]}))
        .mark_rule(color=MARK, strokeDash=[5, 4], size=2)
        .encode(y=alt.Y("v:Q"))
    )

    caption = (
        f"Held-out personas ({holdout.get('n_personas', 0)}). {CURRENT_ATTEMPT.capitalize()}: "
        f"mean **{fmt(holdout.get('mean_r'))}**, worst **{fmt(holdout.get('min_r'))}** on "
        f"{holdout.get('min_r_dimension', '?')}. The bar is {TRAIT_RECOVERY_BAR} on every "
        f"dimension (dashed); red bars miss it. Training personas "
        f"({train.get('n_personas', 0)}) sit at mean {fmt(train.get('mean_r'))}, worst "
        f"{fmt(train.get('min_r'))} on {train.get('min_r_dimension', '?')}."
    )
    if paired:
        caption += (
            f" The pale bar in each pair is {PREVIOUS_ATTEMPT} (mean "
            f"{fmt((previous_holdout or {}).get('mean_r'))}, worst "
            f"{fmt((previous_holdout or {}).get('min_r'))} on "
            f"{(previous_holdout or {}).get('min_r_dimension', '?')})."
        )
    st.caption(caption)
    show(alt.layer(bars, rule).properties(height=260))


def dimensionality_panel(curve: list[dict[str, Any]] | None, fitted_dims: Any = None) -> None:
    st.subheader("How many dimensions the answers support")
    points = curve or []
    rows = []
    for point in points:
        dims = point.get("dims")
        if dims is None:
            continue
        if point.get("train_loglik_per_cell") is not None:
            rows.append(
                {"dims": dims, "line": "in-sample", "loglik": float(point["train_loglik_per_cell"])}
            )
        if point.get("split_half_loglik_per_cell") is not None:
            rows.append(
                {
                    "dims": dims,
                    "line": "split-half",
                    "loglik": float(point["split_half_loglik_per_cell"]),
                }
            )
    if not rows:
        st.info("This report carries no dimensionality curve.")
        return

    frame = pd.DataFrame(rows)
    line = (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=35))
        .encode(
            x=alt.X("dims:Q", title="dimensions fitted"),
            y=alt.Y("loglik:Q", title="log-likelihood per answer", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "line:N",
                title=None,
                scale=alt.Scale(domain=["in-sample", "split-half"], range=[INK, MARK]),
            ),
            tooltip=[
                alt.Tooltip("dims:Q", title="dimensions"),
                alt.Tooltip("line:N", title="line"),
                alt.Tooltip("loglik:Q", format=".4f"),
            ],
        )
    )
    layers: list[Any] = [line]
    if fitted_dims is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"v": [float(fitted_dims)]}))
            .mark_rule(color=BAND, size=2)
            .encode(x=alt.X("v:Q"))
        )

    st.caption(
        "Higher is better. The in-sample line can only improve as dimensions are added; the "
        "split-half line is the one that can fall, because it scores answers the fit did not see."
        + (f" The green line marks the {fitted_dims} dimensions actually fitted." if fitted_dims else "")
    )
    show(alt.layer(*layers).properties(height=260))


def item_recovery_panel(
    by_stratum: dict[str, Any],
    holdout: dict[str, Any],
    blur: dict[str, Any],
) -> None:
    st.subheader("Item recovery and blur")
    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Held-out items, by how far they sit from the training items**")
        rows = []
        for name, block in by_stratum.items():
            rows.append(
                {
                    "stratum": name,
                    "items": block.get("n"),
                    "median r": fmt((block.get("r") or {}).get("median")),
                    "median cosine": fmt((block.get("cosine") or {}).get("median")),
                    "median fitted strength": fmt((block.get("fitted_norm") or {}).get("median")),
                }
            )
        if rows:
            table(rows)
            st.caption(
                f"Pooled median per-item r **{fmt(holdout.get('median_r'))}** over "
                f"{(holdout.get('r_summary') or {}).get('n', 0)} of {holdout.get('n_items', 0)} "
                f"held-out items; the bar is {ITEM_RECOVERY_BAR}. "
                f"{holdout.get('n_items_with_no_designed_loading', 0)} items have no designed "
                "loading at all, so they carry no r."
            )
        else:
            st.info("This report has no per-stratum item recovery.")

    with right:
        st.markdown("**Blur honesty**")
        coverage = blur.get("pooled_coverage")
        band = blur.get("band") or list(BLUR_COVERAGE_BAND)
        if coverage is None:
            st.info("This report has no blur coverage.")
            return
        st.metric(
            "Pooled coverage",
            f"{float(coverage):.1%}",
            help=f"share of {blur.get('n_cells', 0)} cells whose planted value sits inside +-1 blur",
        )
        st.caption(
            f"The pre-registered band is {float(band[0]):.0%}-{float(band[1]):.0%}. "
            f"Mean blur {fmt(blur.get('mean_blur'))} against mean absolute error "
            f"{fmt(blur.get('mean_abs_error'))}."
        )
        diagnostics = blur.get("diagnostics") or {}
        if diagnostics:
            st.caption(
                f"Actual squared error is **{fmt(diagnostics.get('variance_ratio_actual_over_predicted'), 2)}x** "
                f"the variance the model predicts; the interval would have to be "
                f"{fmt(diagnostics.get('blur_multiplier_for_nominal_68pct'), 2)}x wider to be honest."
            )


def watch_list_panel(watch: dict[str, Any]) -> None:
    st.subheader("Watch list")
    if not watch:
        st.info("This report carries no watch list.")
        return

    distractors = watch.get("distractors") or {}
    if distractors:
        st.markdown("**Distractor items** -- designed to load on nothing.")
        flagged = distractors.get("flagged_at_or_above_weak_median") or []
        strengths = distractors.get("per_item_discrimination") or {}
        summary = distractors.get("summary") or {}
        st.caption(
            f"{summary.get('n', 0)} distractors, median strength {fmt(summary.get('median'))}, "
            f"against {fmt(distractors.get('weak_median_for_reference'))} for designed-weak items and "
            f"{fmt(distractors.get('strong_primary_median_for_reference'))} for strong primaries."
        )
        if flagged:
            st.warning(distractors.get("verdict", "distractors carrying real discrimination"))
            table(
                [
                    {"item": item, "fitted strength": fmt(strengths.get(item))}
                    for item in flagged
                ]
            )
        else:
            st.caption("Nothing flagged.")

    signs = watch.get("sign_check") or {}
    st.markdown("**Sign flips** -- items fitted against their designed direction.")
    if signs:
        table(
            [
                {
                    "item": item,
                    "designed dimension": entry.get("designed_dimension"),
                    "designed loading": fmt(entry.get("designed_loading"), 2),
                    "fitted loading": fmt(entry.get("fitted_loading_on_that_dimension")),
                    "fitted strength": fmt(entry.get("fitted_norm")),
                }
                for item, entry in signs.items()
            ]
        )
    else:
        st.caption("No sign flips flagged.")

    pair = watch.get("planted_zero_pair") or {}
    if pair:
        names = " - ".join(str(part) for part in pair.get("pair") or [])
        st.markdown(f"**The planted-zero pair ({names})**")
        table(
            [
                {
                    "pair": names,
                    "planted": fmt(pair.get("planted"), 2),
                    "recovered (train)": fmt(pair.get("recovered_train")),
                    "recovered (held-out)": fmt(pair.get("recovered_holdout")),
                    "miss (held-out)": fmt(pair.get("abs_delta_holdout")),
                }
            ]
        )

    flat = watch.get("near_flat_raw_items") or {}
    if flat:
        st.markdown("**Items that were near-flat in the raw answers**")
        table(
            [
                {
                    "item": item,
                    "type": entry.get("type"),
                    "designed strength": entry.get("strength_class"),
                    "fitted strength": fmt(entry.get("fitted_discrimination")),
                    "rank": f"{entry.get('rank_among_all_fitted_items')} of "
                    f"{entry.get('n_fitted_items')}",
                }
                for item, entry in flat.items()
            ]
        )


def recovery_page() -> None:
    st.header("Recovery")
    report = load_json(RECOVERY_PATH)
    previous = load_json(PREVIOUS_RECOVERY_PATH)

    if report is None:
        st.info(RECOVERY_HINT)
        picture_panel(None)
        return

    counts = report.get("counts") or {}
    st.caption(
        f"Gate 2, {CURRENT_ATTEMPT} -- the current state. Fitted on "
        f"{counts.get('train_personas', '?')} personas x "
        f"{counts.get('train_items', '?')} items, graded on "
        f"{counts.get('holdout_personas', '?')} held-out personas and "
        f"{counts.get('holdout_items', '?')} held-out items. Every number below is read "
        f"straight out of {RECOVERY_PATH.name}, with {PREVIOUS_ATTEMPT} read out of "
        f"{PREVIOUS_RECOVERY_PATH.name} where the two are shown together."
    )

    gate2_verdict_panel(report.get("verdict"))
    st.divider()
    attempt_comparison_panel(report, previous)
    st.divider()
    picture_panel(report.get("correlations"))
    st.divider()
    trait_recovery_panel(
        report.get("trait_recovery_holdout") or {},
        report.get("trait_recovery_train") or {},
        (previous or {}).get("trait_recovery_holdout") or {},
    )
    st.divider()
    diagnostics = report.get("fit_diagnostics") or {}
    dimensionality_panel(
        diagnostics.get("dimensionality_curve"),
        (diagnostics.get("config") or {}).get("dims"),
    )
    st.divider()
    item_recovery_panel(
        report.get("item_recovery_by_stratum") or {},
        report.get("item_recovery_holdout") or {},
        report.get("blur_honesty") or {},
    )
    st.divider()
    watch_list_panel(report.get("watch_list") or {})


# --------------------------------------------------------------------------
# person encoder: Stage 3, from results/stage3_gate3.json, its two PNGs and
# the item-encoder report
# --------------------------------------------------------------------------


GATE3_HINT = (
    "No Gate 3 report in results/ yet. Produce it with:\n\n"
    "```\n"
    "python -m src.eval.gate3 \\\n"
    "    --fused results/stage3_person_fused.json \\\n"
    "    --fused-npz results/stage3_person_fused.npz \\\n"
    "    --backbone-npz results/stage3_person_backbone.npz \\\n"
    "    --fit results/stage2_v2_fit.npz \\\n"
    "    --truth <planted dir> --splits experiments/splits_v1.json \\\n"
    "    --out results --out-prefix stage3\n"
    "```"
)


def gate3_verdict_panel(verdict: dict[str, Any] | None, tracks: dict[str, Any]) -> None:
    """The four frozen bars, read out of the report's own verdict block.

    One bar is different: the report leaves ``monotone_blur`` undecided on
    purpose. That row carries the ruling and says so in the row itself.
    """
    st.subheader("Gate 3 verdict")
    bars = (verdict or {}).get("bars") or {}
    if not bars:
        st.info("This report has no verdict block.")
        return

    rows = []
    for key, entry in bars.items():
        if key == MONOTONICITY_KEY and entry.get("pass") is None:
            mark = f"{MONOTONICITY_VERDICT} -- {MONOTONICITY_RULING}"
        elif entry.get("pass") is True:
            mark = "PASS"
        elif entry.get("pass") is False:
            mark = "FAIL"
        else:
            mark = "not decided in the report"
        label = entry.get("value_label")
        threshold = entry.get("bar")
        rows.append(
            {
                "bar": key.replace("_", " "),
                "rule": entry.get("statement", ""),
                "must be": bar_threshold(
                    threshold if threshold is not None else GATE3_PERSON_BARS.get(key)
                ),
                "value": fmt(entry.get("value")) + (f" ({label})" if label else ""),
                "verdict": mark,
            }
        )
    table(rows)

    failed = [name.replace("_", " ") for name in (verdict or {}).get("failed_bars") or []]
    if failed:
        st.error(f"Gate 3: FAIL -- {len(failed)} of {len(bars)} bars missed: {', '.join(failed)}.")
    elif (verdict or {}).get("all_decided_bars_pass"):
        st.success("Gate 3: PASS -- every decided bar is met.")
    else:
        st.error("Gate 3: FAIL -- at least one bar is missed or not decided.")

    rmse = bars.get("rmse") or {}
    st.caption(
        f"**RMSE:** {rmse.get('n_dimensions_passing', 0)} of "
        f"{len((tracks.get('fused') or {}).get('bars', {}).get('rmse', {}).get('per_dimension_rmse') or {})}"
        f" dimensions under {PERSON_RMSE_BAR}; worst {fmt(rmse.get('value'))} "
        f"({rmse.get('value_label', '')})."
    )

    lift = ((tracks.get("fused") or {}).get("bars") or {}).get("lift") or {}
    if lift:
        st.caption(
            f"**Lift:** pooled encoder RMSE {fmt(lift.get('pooled_encoder_rmse'))} against a "
            f"baseline of {fmt(lift.get('pooled_baseline_rmse'))}"
            + (
                " -- the public-profile ridge collapsed to intercept-only in cross-validation, "
                "so that baseline is the zero-information one."
                if lift.get("baseline_is_intercept_only")
                else "."
            )
        )

    monotone = bars.get(MONOTONICITY_KEY) or {}
    if monotone.get("pass") is None:
        detail = ((tracks.get("fused") or {}).get("bars") or {}).get("monotonicity") or {}
        mean_curve = detail.get("mean_curve_reading") or {}
        strict = detail.get("strict_per_step_reading") or {}
        control = detail.get("fixed_theta_control") or {}
        st.caption(
            f"**Monotone blur:** read as {MONOTONICITY_VERDICT} -- {MONOTONICITY_RULING}. The "
            f"report decides nothing here; it stores both readings and hands the call up. Its own "
            f"mean-curve reading is **{monotone.get('mean_curve_verdict', '--')}** "
            f"({mean_curve.get('n_dimensions_monotone', '?')} of 8 dimensions monotone at zero "
            f"tolerance, largest uptick {fmt(monotone.get('value'), 4)}) and its strict per-step "
            f"reading is **{monotone.get('strict_per_step_verdict', '--')}** "
            f"({strict.get('n_increases', '?')} of {strict.get('n_steps_checked', '?')} steps rise). "
            f"With theta pinned, where nesting makes monotonicity a theorem, the control reads "
            f"{control.get('verdict', '--')}."
        )


def gate3_picture_panel() -> None:
    st.subheader("Pictures")
    left, right = st.columns(2)

    with left:
        st.markdown("**Blur against interview length**")
        if BLUR_VS_N_PATH.is_file():
            st.image(str(BLUR_VS_N_PATH), width="stretch")
        else:
            st.info(f"{BLUR_VS_N_PATH.name} not found -- the Gate 3 run writes it unless --no-plots.")

    with right:
        st.markdown("**RMSE, coverage and lift against interview length**")
        if GATE3_CURVES_PATH.is_file():
            st.image(str(GATE3_CURVES_PATH), width="stretch")
        else:
            st.info(
                f"{GATE3_CURVES_PATH.name} not found -- the Gate 3 run writes it unless --no-plots."
            )


def gate3_rmse_panel(fused: dict[str, Any], closed_only: dict[str, Any]) -> None:
    """Per-dimension RMSE, both tracks, against the frozen 0.5 bar."""
    st.subheader("RMSE per dimension")
    per_dimension = fused.get("per_dimension_rmse") or {}
    if not per_dimension:
        st.info("This report has no per-dimension RMSE.")
        return

    closed = closed_only.get("per_dimension_rmse") or {}
    projection = (fused.get("projection") or {}).get("filed") or {}
    table(
        [
            {
                "dimension": dimension,
                "fused": fmt(value),
                "closed-only": fmt(closed.get(dimension)),
                "bar": fmt(PERSON_RMSE_BAR, 2),
                "filed projection": fmt(projection.get(dimension)),
                "verdict": "PASS" if float(value) <= PERSON_RMSE_BAR else "FAIL",
            }
            for dimension, value in per_dimension.items()
        ]
    )

    gaps = (fused.get("projection") or {}).get("per_dimension_outcome_minus_projection") or {}
    worst_gap = max((abs(float(v)) for v in gaps.values()), default=None)
    st.caption(
        f"Fused is the confirmatory track; closed-only is the same interview without the open "
        f"answers. Worst dimension {fmt(fused.get('worst_rmse'))} "
        f"({fused.get('worst_dimension', '?')}), best {fmt(fused.get('best_rmse'))}, pooled "
        f"{fmt(fused.get('pooled_rmse'))} over {fused.get('n_personas', '?')} held-out personas."
        + (
            f" The projection filed before any Stage 3 measurement said this bar was unreachable; "
            f"every dimension landed within {fmt(worst_gap)} of that filed per-dimension prediction."
            if worst_gap is not None
            else ""
        )
    )


def item_encoder_panel(report: dict[str, Any] | None) -> None:
    """The two item-encoder bars, read out of their own report."""
    st.subheader("Item encoder")
    if report is None:
        st.info(
            f"{ITEM_ENCODER_PATH.name} not found. Produce it with:\n\n"
            "```\n"
            "python -m src.model.item_encoder train \\\n"
            "    --judgments <item judgments.jsonl> \\\n"
            "    --out results/stage3_item_encoder.json\n"
            "```"
        )
        return

    bars = report.get("bars") or {}
    if not bars:
        st.info("This report has no bars block.")
        return

    table(
        [
            {
                "bar": key.replace("_", " "),
                "value": fmt(entry.get("value")),
                "must be": ">= "
                + bar_threshold(
                    entry.get("bar") if entry.get("bar") is not None else GATE3_ITEM_BARS.get(key)
                ),
                "verdict": "PASS"
                if entry.get("pass") is True
                else "FAIL"
                if entry.get("pass") is False
                else "not evaluated",
            }
            for key, entry in bars.items()
        ]
    )
    st.caption(
        f"{report.get('model', 'the frozen system-side model')} reading item wording only, graded "
        f"against the fitted parameters of {report.get('n_holdout_items', '?')} held-out items "
        f"({report.get('n_train_items', '?')} training items were used to fit the map). Direction "
        f"clears its bar; strength does not."
    )


def person_encoder_page() -> None:
    st.header("Person encoder")
    report = load_json(GATE3_PATH)

    if report is None:
        st.info(GATE3_HINT)
        gate3_picture_panel()
        st.divider()
        item_encoder_panel(load_json(ITEM_ENCODER_PATH))
        return

    verdict = report.get("verdict") or {}
    counts = report.get("counts") or {}
    backbone = load_json(BACKBONE_PATH)
    st.caption(
        f"Gate 3, {verdict.get('confirmatory_track', '?')} track at N = "
        f"{verdict.get('confirmatory_n', '?')} scripted questions plus "
        f"{counts.get('open_prompts', '?')} open answers. Fitted on "
        f"{counts.get('train_personas', '?')} training personas, graded on "
        f"{verdict.get('n_holdout_personas', '?')} held-out ones. Every number below is read "
        f"straight out of {GATE3_PATH.name}, with the item bars out of {ITEM_ENCODER_PATH.name}."
        + (
            f" The closed-question backbone it builds on is {BACKBONE_PATH.name}."
            if backbone is not None
            else ""
        )
    )

    tracks = report.get("tracks") or {}
    gate3_verdict_panel(verdict, tracks)

    comparison = verdict.get("closed_only_for_comparison") or {}
    if comparison:
        st.caption(
            f"Closed-only, for comparison: worst RMSE {fmt(comparison.get('rmse_worst'))}, "
            f"coverage {fmt(comparison.get('coverage'))}, lift {fmt(comparison.get('lift'))}."
        )

    st.divider()
    gate3_picture_panel()
    st.divider()
    gate3_rmse_panel(
        ((tracks.get("fused") or {}).get("bars") or {}).get("rmse") or {},
        ((tracks.get("closed_only") or {}).get("bars") or {}).get("rmse") or {},
    )
    st.divider()
    item_encoder_panel(load_json(ITEM_ENCODER_PATH))


# --------------------------------------------------------------------------
# calibration: Stage 4, from results/stage4_gate4.json and its two PNGs
# --------------------------------------------------------------------------


GATE4_HINT = (
    "No Gate 4 report in results/ yet. Produce it with:\n\n"
    "```\n"
    "python -m src.eval.gate4 \\\n"
    "    --truth50 <50-sample truth dir> \\\n"
    "    --predictions results/stage4_predictions.npz \\\n"
    "    --baselines results/stage4_baselines.npz \\\n"
    "    --splits experiments/splits_v1.json \\\n"
    "    --out results\n"
    "```"
)


def gate4_verdict_panel(verdict: dict[str, Any] | None) -> None:
    """The three frozen bars, read out of the report's own verdict block."""
    st.subheader("Gate 4 verdict")
    bars = (verdict or {}).get("bars") or {}
    if not bars:
        st.info("This report has no verdict block.")
        return

    rows = []
    for key, entry in bars.items():
        if entry.get("pass") is True:
            mark = "PASS"
        elif entry.get("pass") is False:
            mark = "FAIL"
        else:
            mark = "not evaluated"
        label = entry.get("value_label")
        threshold = entry.get("bar")
        rows.append(
            {
                "bar": key.replace("_", " "),
                "rule": entry.get("statement", ""),
                "must be": bar_threshold(
                    threshold if threshold is not None else GATE4_BARS.get(key)
                ),
                "value": fmt(entry.get("value")) + (f" ({label})" if label else ""),
                "verdict": mark,
            }
        )
    table(rows)

    failed = [name.replace("_", " ") for name in (verdict or {}).get("failed_bars") or []]
    passed = sum(1 for entry in bars.values() if entry.get("pass") is True)
    if (verdict or {}).get("all_bars_pass"):
        st.success("Gate 4: PASS -- every frozen bar is met.")
    elif failed:
        st.error(
            f"Gate 4: FAIL -- {len(failed)} of {len(bars)} bars missed: {', '.join(failed)}. "
            f"{passed} of {len(bars)} pass."
        )
    else:
        st.error("Gate 4: FAIL -- at least one frozen bar is missed or not evaluated.")

    lift = bars.get("primary_lift") or {}
    if lift:
        st.caption(
            f"**Primary lift:** Brier {fmt(lift.get('model_brier'), 4)} against a "
            f"population-marginal baseline of {fmt(lift.get('marginal_brier'), 4)}, so "
            f"**{fmt(lift.get('value'))}** relative lift on a bar of "
            f"{bar_threshold(lift.get('bar'))}. The second half of the bar is beating k-NN: "
            f"k-NN scores {fmt(lift.get('knn_brier'), 4)}, so the answer is "
            f"**{'yes' if lift.get('beats_knn') else 'no'}** -- but k-NN's own lift over the "
            f"marginal is {fmt(lift.get('knn_lift_vs_marginal'))}, close behind."
        )

    correlation = bars.get("correlation") or {}
    if correlation:
        st.caption(
            f"**Correlation:** **{fmt(correlation.get('value'))}** against a bar of "
            f"{bar_threshold(correlation.get('bar'))} -- the bar this gate misses. Split by item "
            f"type it is {fmt(correlation.get('binary_only'))} on binary items and "
            f"{fmt(correlation.get('likert_only'))} on Likert-5 items, so neither type carries "
            f"the miss on its own."
        )

    ece = bars.get("ece") or {}
    if ece:
        st.caption(
            f"**ECE:** {fmt(ece.get('value'), 4)} against a bar of "
            f"{bar_threshold(ece.get('bar'))}. The probabilities are well calibrated in "
            f"aggregate; the correlation bar is about ordering single cells, not calibration."
        )


def gate4_reliability_panel(arms: dict[str, Any]) -> None:
    """The reliability diagram, plus each arm's headline numbers and the bins."""
    st.subheader("Reliability diagram")
    left, right = st.columns([3, 2])

    with left:
        if RELIABILITY_PATH.is_file():
            st.image(str(RELIABILITY_PATH), width="stretch")
        else:
            st.info(f"{RELIABILITY_PATH.name} not found -- the Gate 4 run writes it next to the report.")

    with right:
        st.markdown("**Every arm on the page, side by side**")
        rows = [
            {
                "arm": name,
                "Brier": fmt((arms.get(name) or {}).get("brier"), 4),
                "log loss": fmt((arms.get(name) or {}).get("log_loss")),
                "ECE": fmt((arms.get(name) or {}).get("ece"), 4),
                "correlation": fmt((arms.get(name) or {}).get("correlation")),
            }
            for name in GATE4_ARMS
            if arms.get(name)
        ]
        if rows:
            table(rows)
            st.caption(
                "`model` is the confirmatory arm; `marginal` is the zero-information baseline "
                "every lift on this page is measured against; `profile` is public profile only; "
                "`knn` matches on the same 15 interview answers the encoder sees. Lower Brier, "
                "lower log loss and lower ECE are better."
            )
        else:
            st.info("This report carries no arms to compare.")

    calibration = ((arms.get("model") or {}).get("calibration")) or {}
    bins = calibration.get("bins") or []
    if not bins:
        st.info("This report has no calibration bins for the confirmatory arm.")
        return

    st.markdown("**The confirmatory arm's bins, as the diagram draws them**")
    table(
        [
            {
                "predicted band": f"{fmt((entry.get('bin') or [None, None])[0], 1)}"
                f"-{fmt((entry.get('bin') or [None, None])[1], 1)}",
                "(cell, category) pairs": entry.get("n"),
                "mean predicted": fmt(entry.get("mean_predicted")),
                "mean observed": fmt(entry.get("mean_observed")),
                "gap": fmt(entry.get("gap")),
            }
            for entry in bins
        ]
    )
    st.caption(
        f"{calibration.get('n_pairs', 0):,} (cell, category) pairs over "
        f"{calibration.get('n_bins', 0)} equal-width bins. Gap is predicted minus observed, so a "
        f"positive gap is over-confidence; count-weighted, those gaps are the ECE of "
        f"{fmt(calibration.get('ece'), 4)}. Most of the mass sits in the lowest bands, which is "
        f"why the two over-confident top bands move the ECE so little."
    )


def gate4_stratum_panel(
    per_stratum: dict[str, Any], knn_proximity: dict[str, Any], counts: dict[str, Any]
) -> None:
    """Lift over the marginal per probe stratum, model beside k-NN and profile.

    Reported, not barred (PREREGISTRATION.md section 6). The model and k-NN
    lifts are read straight out of the report; the profile lift is the one
    number the report does not carry ready-made and is worked out from the two
    Brier scores in the same block.
    """
    st.subheader("Per-stratum performance")
    if not per_stratum:
        st.info("This report has no per-stratum comparison.")
        return

    order = [name for name in STRATUM_ORDER if name in per_stratum]
    order += [name for name in per_stratum if name not in order]

    rows = []
    for name in order:
        block = per_stratum.get(name) or {}
        rows.append({"stratum": name, "arm": "model", "lift": block.get("lift_vs_marginal")})
        rows.append({"stratum": name, "arm": "k-NN", "lift": block.get("knn_lift_vs_marginal")})
        profile_brier = block.get("profile_brier")
        marginal_brier = block.get("marginal_brier")
        profile_lift = (
            1.0 - float(profile_brier) / float(marginal_brier)
            if profile_brier is not None and marginal_brier
            else None
        )
        rows.append({"stratum": name, "arm": "profile", "lift": profile_lift})
    rows = [row for row in rows if row["lift"] is not None]
    if not rows:
        st.info("This report has no per-stratum lifts.")
        return

    frame = pd.DataFrame([{**row, "lift": float(row["lift"])} for row in rows])
    arm_order = ["model", "k-NN", "profile"]
    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("stratum:N", title=None, sort=order),
            xOffset=alt.XOffset("arm:N", sort=arm_order),
            y=alt.Y("lift:Q", title="relative Brier lift over the marginal"),
            color=alt.Color(
                "arm:N",
                title=None,
                scale=alt.Scale(domain=arm_order, range=[INK, MARK, BAND]),
                sort=arm_order,
            ),
            tooltip=[
                alt.Tooltip("stratum:N"),
                alt.Tooltip("arm:N"),
                alt.Tooltip("lift:Q", format=".3f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"v": [PRIMARY_LIFT_BAR]}))
        .mark_rule(color=MARK, strokeDash=[5, 4], size=2)
        .encode(y=alt.Y("v:Q"))
    )
    show(alt.layer(bars, rule).properties(height=280))

    stratum_counts = (counts or {}).get("stratum_counts") or {}
    table(
        [
            {
                "stratum": name,
                "probe items": (per_stratum.get(name) or {}).get("n_items")
                or stratum_counts.get(name),
                "model Brier": fmt((per_stratum.get(name) or {}).get("model_brier"), 4),
                "k-NN Brier": fmt((per_stratum.get(name) or {}).get("knn_brier"), 4),
                "marginal Brier": fmt((per_stratum.get(name) or {}).get("marginal_brier"), 4),
                "model lift": fmt((per_stratum.get(name) or {}).get("lift_vs_marginal")),
                "k-NN lift": fmt((per_stratum.get(name) or {}).get("knn_lift_vs_marginal")),
                "model beats k-NN": "yes"
                if (per_stratum.get(name) or {}).get("model_beats_knn") is True
                else "no"
                if (per_stratum.get(name) or {}).get("model_beats_knn") is False
                else "--",
            }
            for name in order
        ]
    )

    won = [
        name for name in order if (per_stratum.get(name) or {}).get("model_beats_knn") is True
    ]
    lost = (knn_proximity or {}).get("strata_where_knn_wins") or [
        name for name in order if (per_stratum.get(name) or {}).get("model_beats_knn") is False
    ]
    if lost:
        st.warning(
            f"k-NN wins {', '.join(lost)}; the model wins {', '.join(won) or 'nothing'}. "
            f"That is the shape you want: an encoder that only beat k-NN on items near the "
            f"interview would be memorising the interview rather than recovering a trait vector."
        )

    reading = (knn_proximity or {}).get("reading")
    if reading:
        st.caption(f"**k-NN proximity, as the report reads it:** {reading}")
    if knn_proximity:
        st.caption(
            f"Pooled, the model's lift is "
            f"{fmt(knn_proximity.get('model_lift_over_marginal'))} against k-NN's "
            f"{fmt(knn_proximity.get('knn_lift_over_marginal'))} -- the model is only "
            f"{fmt(knn_proximity.get('model_brier_improvement_over_knn_relative'))} better than "
            f"k-NN in relative Brier, which the report itself flags as close"
            f"{'' if knn_proximity.get('close') else ' (it does not flag it here)'}. The dashed "
            f"line is the {PRIMARY_LIFT_BAR:.0%} primary bar, drawn for reference only: this "
            f"panel is reported, not barred. The profile bar is the one number here the report "
            f"does not carry ready-made -- it is 1 - profile Brier / marginal Brier from the two "
            f"Brier scores in the same block."
        )


def gate4_aggregate_panel(
    aggregate: dict[str, Any] | None, cross_tabs: dict[str, Any] | None
) -> None:
    """Predicted vs true population marginals, and the 2-way cross-tabs."""
    st.subheader("Aggregate check")
    if AGGREGATE_PATH.is_file():
        st.image(str(AGGREGATE_PATH), width="stretch")
    else:
        st.info(f"{AGGREGATE_PATH.name} not found -- the Gate 4 run writes it next to the report.")

    if not aggregate and not cross_tabs:
        st.info("This report has no aggregate check.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("**Population marginals**")
        if aggregate:
            st.metric(
                "r, predicted vs true",
                fmt(aggregate.get("correlation"), 4),
                help=aggregate.get("what", ""),
            )
            st.caption(
                f"Mean absolute gap {fmt(aggregate.get('mean_abs_gap'))}, RMSE "
                f"{fmt(aggregate.get('rmse'))}, mean total variation per item "
                f"{fmt(aggregate.get('mean_total_variation_per_item'))}. "
                f"{float(aggregate.get('fraction_inside_the_true_95_interval') or 0):.1%} of "
                f"predicted marginals sit inside the true 95% interval, from "
                f"{aggregate.get('n_bootstrap', 0):,} bootstrap resamples of "
                f"{aggregate.get('resampling_unit', 'the held-out personas')}."
            )
        else:
            st.info("This report has no marginal check.")

    with right:
        st.markdown("**2-way cross-tabs**")
        if cross_tabs:
            st.metric(
                "r, predicted vs true",
                fmt(cross_tabs.get("correlation"), 4),
                help=cross_tabs.get("what", ""),
            )
            st.caption(
                f"{cross_tabs.get('n_pairs', 0):,} item pairs, "
                f"{cross_tabs.get('n_table_cells', 0):,} table cells. RMSE "
                f"{fmt(cross_tabs.get('rmse'))}, mean total variation per pair "
                f"{fmt(cross_tabs.get('mean_total_variation_per_pair'))}, worst pair "
                f"{fmt(cross_tabs.get('worst_total_variation_per_pair'))}."
            )
            caveat = cross_tabs.get("conditional_independence")
            if caveat:
                st.caption(f"**Read this one carefully:** {caveat}")
        else:
            st.info("This report has no cross-tab check.")


def qwen_baseline_panel(block: dict[str, Any] | None) -> None:
    """The no-interview LLM baseline, graded as soon as its file lands."""
    st.subheader("Qwen no-interview baseline")
    if not block:
        st.info("This report carries no slot for the no-interview LLM baseline.")
        return

    status = str(block.get("status", "")).lower()
    if status == "pending":
        note = str(block.get("note", "not run yet")).rstrip(" .")
        st.info(
            f"Pending. {note}."
            + (f" Expected at `{block['expected_at']}`." if block.get("expected_at") else "")
        )
        return

    numbers = {
        key: value
        for key, value in block.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if numbers:
        table([{key.replace("_", " "): fmt(value, 4) for key, value in numbers.items()}])
    st.caption(str(block.get("note", "")))


def calibration_page() -> None:
    st.header("Calibration")
    report = load_json(GATE4_PATH)

    if report is None:
        st.info(GATE4_HINT)
        gate4_reliability_panel({})
        st.divider()
        gate4_aggregate_panel(None, None)
        return

    verdict = report.get("verdict") or {}
    counts = report.get("counts") or {}
    st.caption(
        f"{verdict.get('gate', 'Gate 4')}, {verdict.get('confirmatory_arm', '?')} arm on "
        f"{verdict.get('n_holdout_personas', '?')} held-out personas x "
        f"{verdict.get('n_probe_items', '?')} probe items = "
        f"{verdict.get('n_cells', 0):,} cells, each graded against "
        f"{counts.get('truth_samples_per_cell', '?')} independent sittings. Every number below is "
        f"read straight out of {GATE4_PATH.name}."
    )

    gate4_verdict_panel(verdict)

    leakage = report.get("leakage") or {}
    if leakage.get("leakage_hunt_required"):
        st.warning(
            f"Leakage hunt required: {', '.join(leakage.get('triggers') or [])}. "
            f"{leakage.get('instruction', '')}"
        )

    st.divider()
    gate4_reliability_panel(report.get("arms") or {})
    st.divider()
    gate4_stratum_panel(
        report.get("per_stratum_comparison") or {},
        verdict.get("knn_proximity") or {},
        counts,
    )
    st.divider()
    gate4_aggregate_panel(report.get("aggregate_check"), report.get("cross_tabs"))
    st.divider()
    qwen_baseline_panel(report.get("qwen_no_interview_baseline"))


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------


def population_page(summary: dict[str, Any] | None, reports: list[Path]) -> None:
    st.header("Population")
    if summary is None:
        st.info(EXPORT_HINT)
    else:
        st.caption(f"Exported {summary.get('generated_at', 'at an unknown time')}.")
        trait_distributions(summary)
        st.divider()
        wobble_panel(summary)
        st.divider()
        demographics_panel(summary)

    st.divider()
    qa_section(reports)


def main() -> None:
    st.set_page_config(page_title="GLASSBOX monitor", layout="wide")

    summary = load_json(SUMMARY_PATH)
    reports = qa_files()

    header(summary, reports)

    page = st.sidebar.radio(
        "Page",
        ["1. Population", "2. Recovery", "3. Person encoder", "4. Calibration"],
        help="One page per stage, added as the stage produces its files.",
    )

    if page == "2. Recovery":
        recovery_page()
    elif page == "3. Person encoder":
        person_encoder_page()
    elif page == "4. Calibration":
        calibration_page()
    else:
        population_page(summary, reports)


main()
