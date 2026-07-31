"""EXPLORATORY, quarantined: the same RL interviewer trained on truth instead.

    python -m src.eval.rl_oracle \\
        --config experiments/stage5_rl.json --profile exploratory_oracle \\
        --fit results/stage2_v2_fit.npz \\
        --episodes <rl batch episode cache>.npz \\
        --truth <the rl batch's planted directory> \\
        --out-dir <checkpoint dir> --results results

Owner RULING 2, recorded before any Stage 5 compute:

    Optional, only if cheap: one exploratory truth-reward policy trained on the
    RL batch alone, to measure the deployable-vs-oracle reward gap -- clearly
    labeled, never confirmatory; its training code must live under src/eval (the
    Wall test forbids truth imports anywhere else) and its weights are
    quarantined from every confirmatory artifact.

So this module exists to answer one question: **how much does the deployable
reward cost us?** The confirmatory policy is paid in the system's own declared
uncertainty, which is a proxy. This one is paid in the thing we actually care
about and can never measure in deployment -- distance from planted truth. The
gap between the two policies' truth-side curves is the price of the proxy.

What is identical, and what is not
----------------------------------
Identical: the policy architecture, the trainer, the horizon, the batch size,
the optimiser, the seeds, the episode cache, the personas. The *only* difference
is the reward callable that gets injected:

    confirmatory   r_t = trace(Sigma_{t-1}) - trace(Sigma_t)             - cost
    oracle         r_t = ||theta_hat_{t-1} - theta*||^2
                       - ||theta_hat_t     - theta*||^2                  - cost

Both are "reduction in a squared-error-shaped quantity minus a question cost";
one uses the covariance the system declares, the other the error the system
cannot see. That symmetry is deliberate -- it makes the gap a statement about
the reward and not about two different algorithms.

Frames
------
The posterior's ``theta_hat`` lives in the fit's latent frame and planted theta
lives on the designed axes, so one of them has to be rotated. The rotation is
fitted ONCE, by orthogonal Procrustes on the full-bank posterior over the
training personas, and then held fixed for the whole run
(:func:`latent_frame_truth`). Fixed rather than refitted per step because a
rotation that moves during training would make the reward non-stationary for
reasons that have nothing to do with the policy. The rotation is orthogonal, so
squared error is the same number in either frame once the truth is rotated.

QUARANTINE. The weights this writes are saved with ``arm =
"exploratory_oracle"``, and :func:`src.rl.policy.load_confirmatory` -- the only
loader any confirmatory code path calls -- refuses that arm. The artifacts are
named ``*_exploratory_oracle_*`` so they are also obvious by eye.
``tests/test_rl.py`` asserts the refusal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..model import mirt
from ..model.person_encoder import item_params_from_fit
from ..model.sequential_posterior import solve
from ..rl.policy import ORACLE_ARM, ORACLE_REWARD
from ..rl.reward import RewardContext, RewardFn
from ..rl.train import load_profile
from ..rl.trainer import EpisodeCache, draw_training_curves, train
from .recovery import load_planted_theta, procrustes_rotation

__all__ = [
    "ORACLE_LABEL",
    "ORACLE_REWARD_DEFINITION",
    "latent_frame_truth",
    "make_truth_reward",
    "main",
    "run",
]

ORACLE_LABEL = (
    "EXPLORATORY, owner-approved (RULING 2, recorded before any Stage 5 "
    "compute), and QUARANTINED. This policy's reward was computed from planted "
    "truth, so its weights are truth-shaped and may never enter a confirmatory "
    "result. It exists only to price the deployable reward: the confirmatory "
    "policy is paid in the system's own declared uncertainty, this one in the "
    "error it cannot see, and the difference between their truth-side curves is "
    "the cost of the proxy. Not a frozen bar; not a deployable artifact."
)

ORACLE_REWARD_DEFINITION = (
    "r_t = ||theta_hat_{t-1} - theta*||^2 - ||theta_hat_t - theta*||^2 - "
    "question_cost, with planted theta rotated once into the fit's latent frame "
    "by orthogonal Procrustes on the full-bank posterior over the training "
    "personas and held fixed thereafter. The mirror image of the confirmatory "
    "reward, with realised truth-side squared error in place of declared "
    "posterior variance."
)


def latent_frame_truth(
    codes: np.ndarray,
    params: mirt.ItemParams,
    theta_true: np.ndarray,
    *,
    hessian: str = "analytic",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Planted theta expressed in the fit's latent frame, plus what it cost.

    ``codes`` is one full sitting of every candidate item for the training
    personas: the best estimate of where each persona sits in the latent frame,
    which is what the rotation should be fitted on.
    """
    posterior = solve(np.asarray(codes, dtype=np.int16), params, hessian=hessian)
    rotation = procrustes_rotation(posterior.theta, theta_true)
    rotated = np.asarray(theta_true, dtype=float) @ rotation.T
    residual = posterior.theta @ rotation - theta_true
    diagnostics = {
        "n_personas": int(posterior.theta.shape[0]),
        "full_bank_pooled_rmse": float(np.sqrt(np.mean(residual**2))),
        "note": (
            "the rotation is fitted once on the full-bank posterior and held "
            "fixed for the whole run; it is orthogonal, so squared error is the "
            "same number in either frame"
        ),
    }
    return rotated, diagnostics


def make_truth_reward(theta_true_latent: np.ndarray) -> RewardFn:
    """The oracle reward callable. See :data:`ORACLE_REWARD_DEFINITION`."""
    truth = np.asarray(theta_true_latent, dtype=float)

    def reward(ctx: RewardContext) -> np.ndarray:
        target = truth[np.asarray(ctx.persona_rows, dtype=np.int64)]
        before = np.sum((ctx.theta_before - target) ** 2, axis=1)
        after = np.sum((ctx.theta_after - target) ** 2, axis=1)
        return before - after - float(ctx.question_cost)

    return reward


def run(
    config_path: Path,
    profile: str,
    fit_path: Path,
    episodes_path: Path,
    truth_dir: Path,
    out_dir: Path,
    results_dir: Path,
    *,
    prefix: str = "stage5_rl_exploratory_oracle",
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    config, run_block = load_profile(config_path, profile)
    fit = np.load(fit_path, allow_pickle=False)
    params, _ = item_params_from_fit(fit, block="train")
    cache = EpisodeCache(episodes_path)

    rows = np.arange(cache.n_personas, dtype=np.int64)
    theta_true = load_planted_theta(truth_dir, cache.persona_ids)
    truth_latent, frame = latent_frame_truth(cache.codes[0], params, theta_true)

    report = train(
        cache,
        params,
        config,
        out_dir=out_dir,
        persona_rows=rows,
        reward_fn=make_truth_reward(truth_latent),
        arm=ORACLE_ARM,
        reward_name=ORACLE_REWARD,
        resume=resume,
        verbose=verbose,
    )
    report["label"] = ORACLE_LABEL
    report["reward_definition"] = ORACLE_REWARD_DEFINITION
    report["latent_frame"] = frame
    report["run"] = run_block
    report["inputs"] = {"fit": str(fit_path), "episodes": str(episodes_path)}

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot = draw_training_curves(report, results_dir / f"{prefix}_training.png")
    report["artifacts"]["plot"] = str(plot)
    path = results_dir / f"{prefix}_training.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"  report -> {path}")
        print(f"  plot   -> {plot}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.rl_oracle",
        description=(
            "EXPLORATORY, quarantined: train the Stage 5 interviewer on "
            "truth-side error reduction to price the deployable reward."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="exploratory_oracle")
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--prefix", default="stage5_rl_exploratory_oracle")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    run(
        args.config, args.profile, args.fit, args.episodes, args.truth,
        args.out_dir, args.results,
        prefix=args.prefix, resume=not args.no_resume, verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
