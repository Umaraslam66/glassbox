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

| 2026-08-03 | M1 card rendering: 400 two-pass cards + quality-gate retries (ledger truth from the call cache; includes ≈$0.056 for 34 discarded unbounded-reasoning cards, logged when discarded) | OpenRouter qwen3.7-flash, reasoning on, budget 512 | 995 calls | 1.35M completion tokens (mostly reasoning) | $0.189 |
| 2026-08-03 | M1 reasoning-knob probes (effort / budget tests, 4 calls) | OpenRouter qwen3.7-flash | 4 | ~30k | ≈$0.004 |
| 2026-08-03 | M1 obedience comparison, both arms 60×24 (frozen cap $0.65) | OpenRouter qwen3.7-flash | 2,880 calls | 1.51M completion tokens (arm B reasoning ≈ 99%) | $0.224 |
| 2026-08-03 | M1 bank blind audits ×2 rounds + verification probes | Gemini 3.5 flash-lite | ≈283 | ≈75k in / ≈1.5k out | ≈$0.025 |

| 2026-08-03 | M1 obedience re-run on corrected bank (Ruling 10d; 480 fresh calls + 2,400 cache hits) | OpenRouter qwen3.7-flash | 480 | 264k completion tokens | $0.039 |
| 2026-08-03 | M1 bank audit round 3 + PRC flutter verification (150 calls) | Gemini 3.5 flash-lite | ≈150 | ≈40k in | ≈$0.014 |

| 2026-08-03 | M1 full-bank pre-noise sweep (Ruling 16): 400 travelers × 140 scenarios, reasoning-off, distributions recorded; ledger truth from the call cache | OpenRouter qwen3.7-flash | 56,000 calls | 14M+ prompt tokens, 56k completion tokens (1 per answer), 0 reasoning | **$0.583** (estimate was ~$0.45; overrun from card-length prompts, +30%) |

| 2026-08-03 | M1 round-2 card re-render (Ruling 19e), 400 cards, quality-gate retries included | OpenRouter qwen3.7-flash, reasoning on, budget 512 | 928 calls | 885k completion tokens | $0.130 |
| 2026-08-03 | M1 bank audit round 4 + spot verifications | Gemini 3.5 flash-lite | ≈145 | ≈40k in | ≈$0.013 |
| 2026-08-03 | M1 round-2 sweep (Ruling 19f) — **DIED AT LAUNCH, HTTP 402 insufficient OpenRouter credits**; zero shards, resumable | OpenRouter qwen3.7-flash | 0 completed | 0 | $0.00 |

**Stage M0 total: ≈ $0.0155. Stage M1 so far: ≈ $1.21. Grand total ≈ $1.23.**
No compute used (Leonardo in maintenance). **OpenRouter balance exhausted —
awaiting owner top-up; ~$0.60 completes the Ruling-19 gate.**

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
