"""The proxy-gaming watch on the RL interviewer (owner RULING 2), post hoc.

    python -m src.eval.rl_proxy_watch \\
        --training-report results/stage5_rl_training.json \\
        --fit results/stage2_v2_fit.npz \\
        --episodes <training episode cache>.npz \\
        --backbone results/stage3_person_backbone.npz \\
        --truth <the training batch's planted directory> \\
        --out results --prefix stage5_rl_proxy_watch

Why this is a separate pass, and not a hook in the trainer
----------------------------------------------------------
Owner RULING 2 forbids planted truth from shaping the confirmatory policy's
weights *at all* -- not through the reward, not through a schedule, not through
an early-stopping rule. The cheapest way to make that airtight is architectural:
the training loop never computes a truth-side number, and this module reads the
checkpoints it left on disk afterwards. Training has finished by the time any of
this runs, so there is no channel from these numbers back into any weight, and
the trainer needs no discipline to keep it that way.

What it measures
----------------
For every saved checkpoint, one greedy interview per persona on the same fixed
sitting, and two curves against the number of questions asked:

* **declared blur** -- the mean per-dimension posterior blur, which is the
  system's own statement about how much it does not know. This is (up to the
  cost constant) the quantity the confirmatory reward is made of.
* **truth-side RMSE** -- distance from planted theta after the same orthogonal
  Procrustes alignment Gate 2, Gate 3 and Gate 5 use.

Both are reported as a share of their own starting value, and the **gap**
between those shares is the number RULING 2 asks for:

    gap(N) = error_remaining(N) - blur_remaining(N)

A positive gap means the system has declared away more of its uncertainty than
it has actually removed from its error -- it is more confident than it has
earned. The non-RL strategies already show this (the heuristic's declared blur
falls to about a quarter of where it started by N = 25 while its truth-side
error falls only to about half), so the question for the RL arm is whether
training *on* the blur makes the gap worse. A divergence is a first-class
finding for the report, not a bug.

The reference arms
------------------
The same personas, the same sitting, the info-gain heuristic and random order,
run through :func:`src.eval.gate5.run_episode` -- the confirmatory grader's own
episode runner. Paired, so the comparison is not hostage to which sitting was
drawn.

Arms and quarantine
-------------------
The checkpoint readout uses the generic policy loader, so it can watch the
exploratory oracle-reward arm as well as the confirmatory one; the arm each
checkpoint belongs to is read off the weights and written into the artifact.
The Gate 5 harness path is stricter: it goes through
:func:`src.rl.policy.load_confirmatory`, which refuses anything but the
confirmatory arm.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..interview.episodes import AnswerBank, EpisodeSimulator
from ..interview.strategies import PolicyStrategy
from ..model import mirt
from ..model.person_encoder import item_params_from_fit
from ..model.sequential_posterior import SequentialPosterior
from ..rl.policy import MLPPolicy
from ..rl.trainer import EpisodeCache
from . import gate5
from .recovery import load_planted_theta, procrustes_rotation

__all__ = ["GAP_DEFINITION", "checkpoint_curves", "harness_arms", "main", "watch"]

SCHEMA = "glassbox.stage5.rl_proxy_watch/1"

GAP_DEFINITION = (
    "gap(N) = error_remaining(N) - blur_remaining(N), where blur_remaining is "
    "the mean declared per-dimension posterior blur at N as a share of its value "
    "before any question (the prior blur, 1 by construction), and "
    "error_remaining is the pooled truth-side RMSE at N as a share of its value "
    "before any question. gap > 0 means the system has declared away more of its "
    "uncertainty than it has removed from its error. Same definition as the "
    "non-RL watch in src.eval.gate5.proxy_gaming_watch, so the numbers are "
    "directly comparable."
)

ALIGNMENT_NOTE = (
    "The checkpoint readout scores the TRAINING personas, so its Procrustes "
    "rotation is necessarily fitted on the same personas it scores (in-sample). "
    "That is fine for a training diagnostic -- the question is whether the two "
    "curves diverge, not what the absolute error is -- and it is why no Gate 5 "
    "number is read off this artifact. The held-out grading is done by "
    "src.eval.gate5 with the usual out-of-sample rotation."
)


# --------------------------------------------------------------------------
# one checkpoint
# --------------------------------------------------------------------------


def checkpoint_curves(
    policy_path: str | Path,
    codes: np.ndarray,
    params: mirt.ItemParams,
    theta_true: np.ndarray,
    *,
    horizon: int,
    round_name: str = "",
    hessian: str = "analytic",
) -> dict[str, Any]:
    """Greedy interviews for one checkpoint: declared blur and truth-side RMSE.

    Runs through :class:`src.interview.episodes.EpisodeSimulator`, the same
    simulator every graded strategy runs through, so the policy is asked for
    questions in exactly the way it will be at evaluation. Truth appears only in
    the ``on_step`` callback, which is grader-side code in a grader-side module.
    """
    codes = np.asarray(codes, dtype=np.int16)
    n_personas, n_items = codes.shape
    theta_true = np.asarray(theta_true, dtype=float)

    policy, meta = MLPPolicy.load(policy_path)
    posterior = SequentialPosterior(params, n_personas, hessian=hessian)
    bank = AnswerBank(codes, round_name)
    strategy = PolicyStrategy(n_personas, n_items, policy, name="policy")

    blur: list[float] = []
    rmse: list[float] = []

    def on_step(step: int, state: SequentialPosterior) -> None:
        rotation = procrustes_rotation(state.theta, theta_true)
        error = state.theta @ rotation - theta_true
        blur.append(float(state.blur.mean()))
        rmse.append(float(np.sqrt(np.mean(error**2))))

    result = EpisodeSimulator(bank, posterior, strategy).run(int(horizon), on_step=on_step)

    blur0 = 1.0  # the prior blur, by construction: theta ~ N(0, I)
    rmse0 = float(np.sqrt(np.mean(theta_true**2)))
    n_grid = list(range(1, len(blur) + 1))
    return {
        "policy": str(policy_path),
        "arm": meta["arm"],
        "reward": meta["reward"],
        "iteration": int(meta["extra"].get("iteration", -1)),
        "n_grid": n_grid,
        "declared_blur_by_n": blur,
        "truth_rmse_by_n": rmse,
        "blur_remaining_by_n": [b / blur0 for b in blur],
        "error_remaining_by_n": [r / rmse0 for r in rmse],
        "gap_by_n": [(r / rmse0) - (b / blur0) for b, r in zip(blur, rmse)],
        "declared_blur_final": blur[-1] if blur else None,
        "truth_rmse_final": rmse[-1] if rmse else None,
        "blur_remaining_final": (blur[-1] / blur0) if blur else None,
        "error_remaining_final": (rmse[-1] / rmse0) if rmse else None,
        "gap_final": ((rmse[-1] / rmse0) - (blur[-1] / blur0)) if blur else None,
        "n_distinct_items": int(np.unique(result.chosen).size),
        "n_distinct_first_questions": int(np.unique(result.chosen[0]).size),
        "rmse_at_n0": rmse0,
    }


# --------------------------------------------------------------------------
# the reference arms, through the Gate 5 harness
# --------------------------------------------------------------------------


def harness_arms(
    codes: np.ndarray,
    params: mirt.ItemParams,
    theta_true: np.ndarray,
    kappa: np.ndarray,
    *,
    horizon: int,
    round_name: str,
    policy_path: str | Path | None,
    scripted_columns: Sequence[int],
    strategies: Sequence[str] = ("heuristic", "random"),
    hessian: str = "analytic",
    seed: int = 0,
) -> dict[str, Any]:
    """Run the arms through :func:`src.eval.gate5.run_episode` and summarise.

    Same personas, same sitting for every arm, so the comparison is paired. The
    rows that fit the rotation and the rows that are scored are the same rows
    here -- these are the training personas and this is a development readout,
    which :data:`ALIGNMENT_NOTE` records.
    """
    codes = np.asarray(codes, dtype=np.int16)
    n_personas = codes.shape[0]
    rows = np.arange(n_personas)

    names = list(strategies) + ([gate5.POLICY] if policy_path is not None else [])
    out: dict[str, Any] = {}
    for name in names:
        task = {
            "arm": name,
            "strategy": name,
            "replicate": 0,
            "round": round_name,
            "codes": codes,
            "loadings": params.loadings,
            "cut_raw": params.cut_raw,
            "kinds": list(params.kinds),
            "scripted_columns": list(scripted_columns),
            "seed": seed,
            "theta_true_train": theta_true,
            "theta_true_holdout": theta_true,
            "fit_rows": rows,
            "score_rows": rows,
            "n_train": n_personas,
            "kappa": kappa,
            "n_steps": int(horizon),
            "hessian": hessian,
            "policy_path": str(policy_path) if policy_path is not None else None,
        }
        result = gate5.run_episode(task)
        out[name] = gate5.summarize_arm(
            [result], budget=int(horizon), theta_true_holdout=theta_true
        )
    return out


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def watch(
    training_report: Path,
    fit_path: Path,
    episodes_path: Path,
    backbone_path: Path,
    truth_dir: Path,
    out_dir: Path,
    *,
    prefix: str = "stage5_rl_proxy_watch",
    horizon: int | None = None,
    round_index: int = 0,
    n_personas: int | None = None,
    splits_path: Path | None = None,
    compare: Sequence[str] = ("heuristic", "random"),
    make_plot: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    started = time.time()
    report = json.loads(Path(training_report).read_text(encoding="utf-8"))
    fit = np.load(fit_path, allow_pickle=False)
    backbone = np.load(backbone_path, allow_pickle=False)
    params, item_ids = item_params_from_fit(fit, block="train")
    cache = EpisodeCache(episodes_path)
    horizon = int(horizon or report["config"]["horizon"])

    train_ids = report["training_personas"]["ids"]
    if n_personas is not None:
        train_ids = train_ids[: int(n_personas)]
    rows = cache.rows_for(train_ids)
    codes = cache.codes[int(round_index)][rows]
    round_name = cache.rounds[int(round_index)]
    theta_true = load_planted_theta(truth_dir, train_ids)
    kappa = np.asarray(backbone["blur_kappa"], dtype=float)

    scripted_columns: list[int] = []
    if splits_path is not None:
        splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
        column_of = {iid: j for j, iid in enumerate(item_ids)}
        scripted_columns = [
            column_of[i] for i in (splits.get("interview_closed") or []) if i in column_of
        ]

    if verbose:
        print(
            f"{len(report['checkpoints'])} checkpoints x {rows.size} training "
            f"personas x {horizon} questions on sitting {round_name}"
        )

    curves = []
    for k, checkpoint in enumerate(report["checkpoints"], start=1):
        curves.append(
            checkpoint_curves(
                checkpoint["policy"], codes, params, theta_true,
                horizon=horizon, round_name=round_name,
                hessian=report["config"]["hessian"],
            )
        )
        if verbose:
            row = curves[-1]
            print(
                f"  [{k}/{len(report['checkpoints'])}] iter {row['iteration']:6d}  "
                f"blur left {row['blur_remaining_final']:.3f}  "
                f"error left {row['error_remaining_final']:.3f}  "
                f"gap {row['gap_final']:+.3f}  items {row['n_distinct_items']}"
            )

    final_policy = report["artifacts"]["final_policy"]
    reference = harness_arms(
        codes, params, theta_true, kappa,
        horizon=horizon, round_name=round_name,
        policy_path=final_policy if report["arm"] == "confirmatory" else None,
        scripted_columns=scripted_columns,
        strategies=compare,
        hessian=report["config"]["hessian"],
    )

    out = {
        "schema": SCHEMA,
        "arm": report["arm"],
        "reward": report["reward"],
        "gap_definition": GAP_DEFINITION,
        "alignment": ALIGNMENT_NOTE,
        "inputs": {
            "training_report": str(training_report),
            "episodes": str(episodes_path),
            "fit": str(fit_path),
            "backbone": str(backbone_path),
        },
        "design": {
            "n_personas": int(rows.size),
            "horizon": horizon,
            "sitting": round_name,
            "n_checkpoints": len(curves),
            "personas": "the policy's own TRAINING personas -- never the confirmatory held-out set",
        },
        "checkpoints": curves,
        "gate5_harness_arms": {
            name: {
                "pooled_rmse_mean_by_n": arm["pooled_rmse_mean_by_n"],
                "rmse_worst_dim_mean_by_n": arm["rmse_worst_dim_mean_by_n"],
                "declared_blur_mean_by_n": arm["declared_blur_mean_by_n"],
                "rmse_at_n0": arm["rmse_at_n0"],
                "pooled_rmse_full_bank": arm["pooled_rmse_full_bank"],
                "fraction_of_full_matrix": arm["fraction_of_full_matrix"],
                "mean_distinct_items_used": arm["mean_distinct_items_used"],
            }
            for name, arm in reference.items()
        },
        "gate5_harness_watch": gate5.proxy_gaming_watch(reference),
        "costs": {
            "wall_seconds": round(time.time() - started, 1),
            "gpu": "none",
            "api": "none",
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}.json"
    if make_plot:
        out["artifacts"] = {"plot": str(draw_watch(out, out_dir / f"{prefix}.png"))}
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"  report -> {path}")
    return out


def draw_watch(report: dict[str, Any], path: Path) -> Path:
    """Two panels: the gap as training proceeds, and the two curves at the end."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = report["checkpoints"]
    iterations = [c["iteration"] for c in curves]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    ax = axes[0]
    ax.plot(iterations, [c["blur_remaining_final"] for c in curves],
            color="#C44E52", lw=1.8, marker="o", ms=3, label="declared blur left")
    ax.plot(iterations, [c["error_remaining_final"] for c in curves],
            color="#4C72B0", lw=1.8, marker="o", ms=3, label="truth-side error left")
    ax.plot(iterations, [c["gap_final"] for c in curves],
            color="#55A868", lw=1.4, ls="--", label="gap")
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.set_xlabel("training iteration")
    ax.set_ylabel(f"share of the starting value left after {report['design']['horizon']} questions")
    ax.set_title("does training on blur make the gap worse?")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    last = curves[-1]
    ax.plot(last["n_grid"], last["blur_remaining_by_n"], color="#C44E52", lw=1.8,
            label="policy: blur left")
    ax.plot(last["n_grid"], last["error_remaining_by_n"], color="#C44E52", lw=1.8,
            ls="--", label="policy: error left")
    colours = {"heuristic": "#8172B2", "random": "#937860", "policy": "#C44E52"}
    for name, arm in report["gate5_harness_watch"].items():
        if name == "policy":
            continue
        grid = [row["n"] for row in arm["by_n"]]
        ax.plot(grid, [row["blur_remaining"] for row in arm["by_n"]],
                color=colours.get(name, "#888888"), lw=1.4, label=f"{name}: blur left")
        ax.plot(grid, [row["error_remaining"] for row in arm["by_n"]],
                color=colours.get(name, "#888888"), lw=1.4, ls="--",
                label=f"{name}: error left")
    ax.axhline(1.0, color="#999999", lw=0.8)
    ax.set_xlabel("N closed questions asked")
    ax.set_ylabel("share of the starting value still there")
    ax.set_title("the trained policy against the non-RL strategies, same sitting")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    fig.suptitle(
        f"Stage 5 RL proxy-gaming watch (owner RULING 2) -- {report['arm']} arm, "
        f"training personas, DIAGNOSTIC ONLY",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.rl_proxy_watch",
        description=(
            "Post-hoc proxy-gaming watch over saved RL checkpoints (owner RULING 2). "
            "Truth-side numbers are recorded for the report and reach no weight."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", default="stage5_rl_proxy_watch")
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--personas", type=int, default=None)
    parser.add_argument("--compare", default="heuristic,random")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    watch(
        args.training_report, args.fit, args.episodes, args.backbone, args.truth,
        args.out,
        prefix=args.prefix,
        horizon=args.horizon,
        round_index=args.round_index,
        n_personas=args.personas,
        splits_path=args.splits,
        compare=[s.strip() for s in args.compare.split(",") if s.strip()],
        make_plot=not args.no_plot,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
