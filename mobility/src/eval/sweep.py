"""Full-bank pre-noise sweep (vault-side): every traveler answers every scenario.

Authorized by Gate M1 Ruling 12, contingent on the obedience re-run passing
the frozen rule. Reasoning-off (the frozen answering regime), one round,
answer-token distributions recorded for the noise layer — request/response
field names come from ``mobility/config/api_fields.json`` per Gate M0
Ruling 4. Everything this sweep produces (pre-noise answers, distributions)
is truth material and stays in the vault; the noise layer's output is what
the system side will ever see.

Resumable per traveler: a traveler with a complete shard is never re-swept,
and the API cache below that never re-queries. Run:
``python -m mobility.src.eval.sweep mobility/experiments/m1_sweep.json``
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mobility.src.engine.or_client import ORClient
from mobility.src.eval.truth_vault import truth_dir

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"

_ANSWER_SYSTEM_SUFFIX = (
    "\n\nAnswer every question fully in character, based on how this person "
    "would actually choose. Reply with exactly one letter and nothing else."
)


def main(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    vault = truth_dir()
    run_dir = vault / "runs" / config["experiment"]
    shard_dir = run_dir / "answers"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Ruling-4 read site: distribution field names live in config, not code.
    fields = json.loads(
        (MOBILITY_ROOT / "config" / "api_fields.json").read_text())["openrouter"]

    bank = {s["sid"]: s for s in json.loads(
        (MOBILITY_ROOT / "data" / "bank" / "scenarios.json").read_text())}
    sids = sorted(bank)
    card_dir = vault / "runs" / "m1_cards" / "cards"
    ids = [f"t{i:04d}" for i in range(400)]
    cards = {pid: json.loads((card_dir / f"{pid}.json").read_text())["card"]
             for pid in ids}

    client = ORClient(cache_dir=run_dir / "cache")
    temperature = float(config["temperature"])

    def sweep_traveler(pid: str) -> None:
        shard = shard_dir / f"{pid}.jsonl"
        if shard.is_file() and len(shard.read_text().splitlines()) == len(sids):
            return
        rows = []
        for sid in sids:
            record = client.call(
                cards[pid] + _ANSWER_SYSTEM_SUFFIX, bank[sid]["text"],
                temperature=temperature, max_tokens=8, reasoning_on=False,
                seed_key=f"{pid}:{sid}:sweep:r0", dist_fields=fields)
            letter = (record["text"] or "?").strip()[:1].upper()
            valid = {o["key"] for o in bank[sid]["options"]}
            rows.append({"pid": pid, "sid": sid,
                         "answer": letter if letter in valid else None,
                         "dist_top": record.get("dist_top")})
        with shard.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    done_before = sum(1 for pid in ids if (shard_dir / f"{pid}.jsonl").is_file())
    with ThreadPoolExecutor(max_workers=int(config["concurrency"])) as pool:
        for k, _ in enumerate(pool.map(sweep_traveler, ids)):
            if (k + 1) % 25 == 0:
                print(f"{k + 1}/400 travelers swept; client: {client.stats()}",
                      flush=True)

    shards = [json.loads(line)
              for pid in ids
              for line in (shard_dir / f"{pid}.jsonl").read_text().splitlines()]
    n_null = sum(1 for r in shards if r["answer"] is None)
    n_dist = sum(1 for r in shards if r.get("dist_top"))
    summary = {
        "experiment": config["experiment"],
        "date": _dt.date.today().isoformat(),
        "n_travelers": 400, "n_scenarios": len(sids),
        "n_answers": len(shards),
        "n_invalid_answers": n_null,
        "share_with_distribution": round(n_dist / len(shards), 4),
        "travelers_already_done_at_start": done_before,
        "client": client.stats(),
        "config": {k: v for k, v in config.items()},
    }
    (RESULTS_DIR / f"{config['experiment']}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = main(Path(sys.argv[1]))
    print(json.dumps({k: out[k] for k in
                      ("n_answers", "n_invalid_answers", "share_with_distribution",
                       "client")}, indent=2))
