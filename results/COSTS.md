# Costs

Every experiment logs its compute and API cost here, as it runs, not afterwards.

**Warning on OpenRouter `qwen/qwen3.7-flash`:** it is a reasoning model and spends
hidden thinking tokens on every call (~180 output tokens for a one-word reply).
Its effective per-call cost is several times the headline per-token price. Budget
accordingly if it is ever used for bulk development calls.

| Date | Stage | What | Compute/API | Cost |
| --- | --- | --- | --- | --- |
| 2026-07-30 | 0 | Gemini API path verification (1 trivial call) | gemini-3.5-flash-lite API, 8 in / 1 out tokens | ≈$0.00 |
| 2026-07-30 | 0 | OpenRouter fallback verification (1 trivial call) | qwen/qwen3.7-flash API, 17 in / 178 out tokens (reasoning model) | ≈$0.00 |
| 2026-07-30 | 0 | Gemma smoke test on Leonardo (job 51068113, 1 node × 10m46s, boost_qos_dbg) | 5.74 core-hours ([redacted-allocation]) | 5.74 core-h |
| 2026-07-30 | 1 | Question bank generation (252 closed + 20 open items, 108 drafting calls) | gemini-3.5-flash-lite API, ~44k in tokens | ≈$0.02 |
| 2026-07-30 | 1 | Qwen3.6-27B copy [sibling-project] → glassbox (login-node I/O, no jobs) | none billed | 0 |
| 2026-07-31 | 1 | Gemma card writing, 500 personas (job 51146579, 1 node × 11m17s) | 6.02 core-hours | 6.02 core-h |
| 2026-07-31 | 1 | Card retry for 6 leak-rejected personas (job 51150672, 1 node × 9m26s) | 5.03 core-hours | 5.03 core-h |
| 2026-07-31 | 1 | Temperature pilot, attempt 1 — p0314 card rendered, both answer sweeps died on leaked GPU memory (job 51171292, 1 node × 10m40s) | 5.69 core-hours | 5.69 core-h |
| 2026-07-31 | 1 | Temperature pilot, attempt 2 — 60 personas × (252 main + 30 retest) at t=0.7 and t=1.0, 33,840 answers (job 51179813, 1 node × 16m42s) | 8.91 core-hours | 8.91 core-h |
| 2026-07-31 | 1 | Temperature pilot round 2 (clean per-record seeds) — 3 arms t=0.7/1.0/1.3, 50,760 answers, one engine init (job 51189257, 1 node × 13m37s) | 7.26 core-hours | 7.26 core-h |
| 2026-07-31 | 1 | Re-render t=0.7 arm recording top-20 answer-token logprobs, input to the mechanical noise layer (job 51194936, 1 node × 10m07s) | 5.40 core-hours | 5.40 core-h |
| 2026-07-31 | 1 | Noise-layer tuning grid (375 configs, offline on the recorded logprobs) | local CPU, no GPU | 0 |
| 2026-07-31 | 1 | FULL SWEEP — 500 personas × (252 main + 30 retest) at t=0.7 with logprobs; 16,920 cells reused from job 51194936, 124,080 rendered (job 51200138, 1 node × 20m09s) | 10.75 core-hours | 10.75 core-h |
| 2026-07-31 | 2 | MIRT fit — 3 restarts at d=8 plus the d ∈ {2..12} dimensionality curve, held-out persona scoring and held-out item fitting (local, numpy, single core) | local CPU, 59s wall, no GPU, no API | 0 |
| 2026-07-31 | 2 | Recovery grading, plots and diagnostics | local CPU, 2s wall | 0 |
| 2026-07-31 | 2-fix | Card regeneration v2 — TWO-PASS (write + self-check/revise), 500 personas, both passes + bridge in one job (job 51317905, 1 node × 15m00s) | 8.00 core-hours | 8.00 core-h |
| 2026-07-31 | 2-fix | FULL SWEEP v2 on the regenerated cards — 500 personas × (252 main + 30 retest) at t=0.7 with top-20 logprobs, 141,000 answers, nothing reused (job 51324033, 1 node × 19m43s, boost_qos_dbg after a queue-congestion resubmit) | 10.52 core-hours | 10.52 core-h |
| 2026-07-31 | 2-fix | Gate 1 v2 QA pipeline — parse, frozen noise layer, population QA on noised and raw answers | local CPU, ~10s wall, no GPU, no API | 0 |
| 2026-07-31 | 2-fix | MIRT fit v2 on the regenerated cards — same frozen settings as attempt 1 (d=8, ridge 0.01, seeds 0/1/2, max-iters 6000, dimensionality curve d ∈ {2..12}) | local CPU, 57.9s wall, no GPU, no API | 0 |
| 2026-07-31 | 2-fix | Recovery grading v2 — two passes (confirmatory, then again with --exclude-items q229,q234 for the robustness block), plots and diagnostics | local CPU, 2.7s wall total, no GPU, no API | 0 |
| 2026-07-31 | 2-fix | Gate 2 second-attempt analysis — correlation-matrix comparison, model-free TRU–RSK, exploratory blur decomposition (read-only over existing artifacts) | local CPU, ~30s wall, no GPU, no API | 0 |
| 2026-07-31 | 3 | Scripted interview closed answers — 500 personas × 15 frozen items drawn from the v2 sweep's recorded material by the frozen noise layer (round tag interview1); no model call, the responder is never re-queried | local CPU, 1.5s wall, no GPU, no API | 0 |
| 2026-07-31 | 3 | Stage 3 transcript assembly (closed part) — 500 transcripts, 7,500 turns | local CPU, ~2s wall, no GPU, no API | 0 |
| 2026-07-31 | 3 | Open-ended interview answers — 500 personas × 3 frozen open prompts = 1,500 generations at t=0.7, max 250 tokens (job 51333926, 1 node × 8m02s, boost_qos_dbg) | 4.28 core-hours | 4.28 core-h |
| 2026-07-31 | 3 | Person-encoder backbone — MAP/Laplace posterior over the 15 scripted closed answers for all 500 personas at every prefix N = 1..15, blur calibration on training residuals, public-profile-only ridge baseline (no LLM, no GPU) | local CPU, 1.0s wall, no GPU, no API | 0 |
