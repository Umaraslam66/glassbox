"""GLASSBOX monitoring dashboard -- page 1 Population, page 2 Recovery.

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
        ["1. Population", "2. Recovery"],
        help="One page per stage, added as the stage produces its files.",
    )

    if page == "2. Recovery":
        recovery_page()
    else:
        population_page(summary, reports)


main()
