"""Gate M0 Ruling 6: the committed pilot tool reproduces the M0 record's shape.

Two layers:

* offline (always runs): ``build_summary`` fed canned normalized records must
  produce the same fields as the Stage M0 record
  (``mobility/results/m0_pilot_summary.json``) with sane values, and the
  distribution verdicts must derive correctly from the records.
* live smoke (opt-in, spends ~$0.002): set ``GLASSBOX_M0_SMOKE=1`` to run the
  real ``--small`` pilot against both providers and hold its output to the
  same shape checks. Never run implicitly on a commit.

One deliberate difference, stated rather than hidden: the M0 record's
repeat-sample block was measured reasoning-on; the tool's repeat block runs
the approved answering regime (reasoning-off), so the repeat-table field is
matched by prefix, not by exact name.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mobility.tools.m0_pilot import build_summary, run_pilot

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD = json.loads(
    (REPO_ROOT / "mobility" / "results" / "m0_pilot_summary.json").read_text()
)


def canned_or_rows() -> list[dict]:
    rows = [
        {"provider": "openrouter", "tag": "probe", "status": 200, "latency_s": 9.1,
         "answer": "A", "prompt_tokens": 252, "completion_tokens": 1101,
         "reasoning_tokens": 1095, "cost_usd": 1.5e-4,
         "dist_top": [["A", 1.0], ["B", 0.0]], "dist_requested": True,
         "reasoning_off": False},
        {"provider": "openrouter", "tag": "probe_off", "status": 200, "latency_s": 0.5,
         "answer": "B", "prompt_tokens": 254, "completion_tokens": 1,
         "reasoning_tokens": 0, "cost_usd": 7.7e-6,
         "dist_top": [["B", 0.58], ["A", 0.31], ["C", 0.10]], "dist_requested": True,
         "reasoning_off": True},
        {"provider": "openrouter", "tag": "burst", "status": 429, "latency_s": 0.4,
         "error": "upstream shared pool", "retry_after": None},
    ]
    for letter in ("A", "B", "B"):
        rows.append({"provider": "openrouter", "tag": "repeat_s1", "status": 200,
                     "latency_s": 0.5, "answer": letter, "prompt_tokens": 254,
                     "completion_tokens": 1, "reasoning_tokens": 0,
                     "cost_usd": 7.7e-6, "dist_top": [[letter, 0.6]],
                     "dist_requested": True, "reasoning_off": True})
    return rows


def canned_gem_rows() -> list[dict]:
    return [
        {"provider": "gemini", "tag": "probe", "status": 400, "latency_s": 0.4,
         "error": "{'message': 'that feature is not enabled for this model'}",
         "dist_requested": True},
        {"provider": "gemini", "tag": "seq", "status": 200, "latency_s": 0.5,
         "answer": "A", "prompt_tokens": 246, "completion_tokens": 1,
         "reasoning_tokens": 0, "dist_top": None, "dist_requested": False},
    ]


def canned_summary() -> dict:
    return build_summary(
        canned_or_rows(), canned_gem_rows(),
        leonardo={"checked": "2026-08-03T15:00+02:00", "result": "test", "verdict": "not checked"},
        run_label="canned",
    )


def assert_record_shape(summary: dict) -> None:
    """The shape contract: every field of the M0 record, sane values, no numbers compared."""
    for key in RECORD:
        assert key in summary, f"missing top-level field {key!r}"

    ors = summary["openrouter_qwen37_flash"]
    for key in ("requests_total", "reasoning_on", "reasoning_off", "rate_limit_nature"):
        assert key in ors, f"missing openrouter field {key!r}"
    assert any(k.startswith("repeat_sample_disagreement") for k in ors), "missing repeat table"
    for regime in ("reasoning_on", "reasoning_off"):
        for key in ("usd_per_1k_scenario_answers", "answer_token_distribution"):
            assert key in ors[regime], f"missing {regime} field {key!r}"

    gem = summary["gemini_35_flash_lite"]
    for key in ("requests_total", "latency_median_s", "answer_token_distribution",
                "pricing_usd_per_m", "usd_per_1k_scenario_answers"):
        assert key in gem, f"missing gemini field {key!r}"

    assert isinstance(summary["costs_usd"]["total"], (int, float))
    assert summary["costs_usd"]["total"] >= 0
    assert summary["leonardo"].get("verdict")


def test_record_itself_carries_the_contract_fields() -> None:
    """Guard: if the M0 record's shape ever changes, this suite must notice."""
    for key in ("experiment", "date", "leonardo", "openrouter_qwen37_flash",
                "gemini_35_flash_lite", "costs_usd"):
        assert key in RECORD


def test_canned_summary_reproduces_the_record_shape() -> None:
    assert_record_shape(canned_summary())


def test_distribution_verdicts_derive_from_the_records() -> None:
    summary = canned_summary()
    ors = summary["openrouter_qwen37_flash"]
    assert ors["reasoning_on"]["answer_token_distribution"].startswith("DEGENERATE")
    assert ors["reasoning_off"]["answer_token_distribution"].startswith("USABLE")
    assert summary["gemini_35_flash_lite"]["answer_token_distribution"].startswith(
        "NOT AVAILABLE")


def test_sane_values_in_canned_summary() -> None:
    summary = canned_summary()
    ors = summary["openrouter_qwen37_flash"]
    assert ors["reasoning_on"]["latency_median_s"] > ors["reasoning_off"]["latency_median_s"]
    assert ors["reasoning_on"]["usd_per_1k_scenario_answers"] > \
        ors["reasoning_off"]["usd_per_1k_scenario_answers"]
    assert ors["requests_total"] == len(canned_or_rows())
    table = ors["repeat_sample_disagreement"]
    assert table.get("s1") == {"A": 1, "B": 2}


@pytest.mark.skipif(os.environ.get("GLASSBOX_M0_SMOKE") != "1",
                    reason="live smoke spends API money; set GLASSBOX_M0_SMOKE=1 to run")
def test_live_small_pilot_reproduces_the_record_shape() -> None:
    from src.llm_client import load_dotenv_if_present

    load_dotenv_if_present(REPO_ROOT / ".env")
    summary = run_pilot(small=True, skip_leonardo=True)
    assert_record_shape(summary)
    ors = summary["openrouter_qwen37_flash"]
    assert ors["requests_total"] >= 10
    assert summary["gemini_35_flash_lite"]["requests_total"] >= 3
    assert 0 < summary["costs_usd"]["total"] < 0.05
