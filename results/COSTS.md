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
