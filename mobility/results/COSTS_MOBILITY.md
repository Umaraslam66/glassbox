# COSTS — GLASSBOX-Mobility

Every experiment's spend, logged at run time, never retroactively.
Compute (Leonardo core-hours) joins when the cluster returns from maintenance.

| Date | Experiment | Provider / model | Requests | Tokens (in / out, of which reasoning) | Cost |
|---|---|---|---|---|---|
| 2026-08-03 | M0 availability pilot (main) | OpenRouter qwen/qwen3.7-flash | 64 (44 ok, 20 rate-limited) | ~11k in / ~64k out, ≈99% of output = hidden reasoning | $0.0086 (provider-metered) |
| 2026-08-03 | M0 availability pilot (main) | Gemini 3.5 flash-lite (AI Studio) | 19 (19 ok) | ~4.7k in / ~19 out, no thinking tokens | ≈$0.0015 (computed at $0.30/M in, $2.50/M out) |
| 2026-08-03 | M0 follow-up probe | OpenRouter qwen/qwen3.7-flash | 32 (32 ok incl. retries) | see `m0_pilot_summary.json` | $0.0034 (provider-metered) |

| 2026-08-03 | Ruling-6 live smoke of the committed pilot tool (`mobility/tools/m0_pilot.py --small` path via `GLASSBOX_M0_SMOKE=1`) | OpenRouter qwen3.7-flash + Gemini flash-lite | 14 + 4 | small run; shape check only | ≈$0.002 |

| 2026-08-03 | M1 population mint (6 seeded attempts, laptop CPU seconds) | none — pure NumPy | 0 API requests | — | $0.00 |

**Stage M0 total: ≈ $0.0155. Stage M1 so far: $0.00.** No compute used
(Leonardo in maintenance).

Notes recorded at run time:

- qwen3.7-flash with default settings spends ~1,300–1,700 hidden reasoning
  tokens per one-letter scenario answer — ≈99% of billed output. Measured
  $0.18–0.22 per 1k scenario answers with reasoning on, ≈$0.008 per 1k with
  reasoning disabled.
- OpenRouter 429s are upstream shared-pool limits (Alibaba side), not our
  key's quota: no Retry-After header, and bursts at concurrency 8 lost 8/24
  requests. Sustainable operation needs low concurrency + backoff + the
  resumable driver (PRD §7 already mandates all three).
- Gemini 3.5 flash-lite: ≈$0.076 per 1k scenario answers at the measured
  246-in/1-out token profile; no rate-limit hits at concurrency 4.
