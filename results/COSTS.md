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
