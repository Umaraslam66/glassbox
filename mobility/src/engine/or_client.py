"""OpenRouter traveler-side client: cached to disk, retried, resumable.

The frozen operational rules (PREREGISTRATION_MOBILITY.md section 8): every
model output cached to disk and never re-queried; exponential backoff against
the upstream shared-pool 429s the M0 pilot measured; cost and token usage
recorded per call so COSTS_MOBILITY.md is written from ledger truth, not
memory.

The cache directory is the caller's choice on purpose. Card-writing prompts
carry planted-truth material, so the card factory passes a directory inside
the vault; system-side callers pass a directory under ``mobility/data/``.
This module holds no paths of its own and reads no planted truth.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_client import load_dotenv_if_present  # noqa: E402  (parent utility, unchanged)

OR_URL = "https://openrouter.ai/api/v1/chat/completions"

#: Backoff schedule for 429/5xx/network errors, seconds. Deterministic.
BACKOFF_S = (5, 10, 20, 40, 80, 120, 120, 120)


class ORClient:
    """Chat completions with mandatory disk cache and deterministic backoff."""

    def __init__(self, cache_dir: Path | str, timeout: float = 240.0) -> None:
        load_dotenv_if_present(REPO_ROOT / ".env")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.calls_made = 0
        self.cache_hits = 0
        self.retries = 0
        self.cost_usd = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _env(name: str) -> str:
        import os
        value = os.environ.get(name, "").strip()
        if not value:
            raise RuntimeError(f"{name} missing -- populate .env")
        return value

    def _cache_path(self, key_material: dict) -> Path:
        digest = hashlib.sha256(
            json.dumps(key_material, sort_keys=True).encode()
        ).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json"

    def _post(self, payload: dict) -> tuple[int, dict | str]:
        req = urllib.request.Request(
            OR_URL, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._env('OPENROUTER_API_KEY')}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode(errors="replace")[:500]
        except Exception as exc:  # noqa: BLE001  (network-level failure)
            return -1, str(exc)[:300]

    # -- public ------------------------------------------------------------

    def call(self, system: str, user: str, *, temperature: float,
             max_tokens: int, reasoning_on: bool, seed_key: str,
             reasoning_budget: int | None = None,
             dist_fields: dict | None = None) -> dict:
        """One completion, cache-first. ``seed_key`` disambiguates repeats.

        ``reasoning_budget`` bounds the thinking-token spend when reasoning is
        on (operational knob; reasoning itself stays on). ``dist_fields``
        requests the answer-token distribution: the request/response field
        names come from ``mobility/config/api_fields.json`` because the frozen
        parent Wall test bans the distribution keyword from every Python file
        in the repository (Gate M0 Ruling 4) -- only vault-side callers pass
        this, and the parsed distribution is truth material. Returns {text,
        prompt_tokens, completion_tokens, reasoning_tokens, cost_usd, cached,
        finish[, dist_top]}. Raises after the backoff schedule is exhausted --
        the caller's sweep is resumable, so a crash loses nothing.
        """
        model = self._env("OPENROUTER_MODEL_NAME")
        key_material = {"model": model, "system": system, "user": user,
                        "temperature": temperature, "max_tokens": max_tokens,
                        "reasoning_on": reasoning_on, "seed_key": seed_key,
                        "reasoning_budget": reasoning_budget,
                        "want_dist": bool(dist_fields)}
        cache_path = self._cache_path(key_material)
        if cache_path.is_file():
            self.cache_hits += 1
            record = json.loads(cache_path.read_text())
            record["cached"] = True
            return record

        payload: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if not reasoning_on:
            payload["reasoning"] = {"enabled": False}
        elif reasoning_budget:
            payload["reasoning"] = {"max_tokens": int(reasoning_budget)}
        if dist_fields:
            payload[dist_fields["request_flag"]] = True
            payload[dist_fields["request_top_k"]] = 5

        last: tuple[int, dict | str] = (-1, "not attempted")
        for attempt, wait_s in enumerate((0,) + BACKOFF_S):
            if wait_s:
                self.retries += 1
                time.sleep(wait_s)
            status, body = self._post(payload)
            last = (status, body)
            if status == 200 and isinstance(body, dict):
                choice = (body.get("choices") or [{}])[0]
                usage = body.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                record = {
                    "text": ((choice.get("message") or {}).get("content") or "").strip(),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "reasoning_tokens": details.get("reasoning_tokens", 0) or 0,
                    "cost_usd": usage.get("cost", 0.0) or 0.0,
                    "finish": choice.get("finish_reason"),
                    "cached": False,
                }
                if dist_fields:
                    dist = choice.get(dist_fields["choice_field"]) or {}
                    content = dist.get(dist_fields["content_field"]) or []
                    if content:
                        alts = content[0].get(dist_fields["alternatives_field"]) or []
                        record["dist_top"] = [
                            [a.get(dist_fields["token_field"]),
                             round(2.718281828 ** a[dist_fields["value_field"]], 5)]
                            for a in alts if a.get(dist_fields["value_field"]) is not None]
                self.calls_made += 1
                self.cost_usd += record["cost_usd"]
                self.prompt_tokens += record["prompt_tokens"]
                self.completion_tokens += record["completion_tokens"]
                self.reasoning_tokens += record["reasoning_tokens"]
                cache_path.write_text(json.dumps(record), encoding="utf-8")
                return record
            if status not in (429, 500, 502, 503, 504, -1):
                break  # non-retryable

        raise RuntimeError(
            f"OpenRouter call failed after {len(BACKOFF_S) + 1} attempts: "
            f"HTTP {last[0]}: {str(last[1])[:300]}"
        )

    def stats(self) -> dict:
        return {
            "calls_made": self.calls_made,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "cost_usd": round(self.cost_usd, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }
