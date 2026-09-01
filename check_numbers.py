"""Fail if any figure quoted in README.md disagrees with results/search_log.csv.

What this covers: the figures listed in claims() are recomputed from the log and
then required to appear in the README with enough surrounding words that the
match cannot be incidental. What it does not cover: claims written in words
rather than digits, such as which operator was wasteful or what a result means.
Those still need reading. This is a drift guard on the arithmetic, not a proof
that the prose is true.
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARAM_CAP = 50_000


def load() -> list[dict]:
    rows = list(csv.DictReader((ROOT / "results" / "search_log.csv").open()))
    for r in rows:
        for k in ("params", "macs", "peak_act", "deployable"):
            r[k] = int(r[k])
        for k in ("acc", "fitness", "train_s"):
            r[k] = float(r[k])
    return rows


def claims(rows: list[dict]) -> list[tuple[str, str]]:
    """(what it is, the exact string the README must contain)."""
    ok = [r for r in rows if r["deployable"]]
    seed, best = rows[0], max(ok, key=lambda r: r["fitness"])
    rejected = [r for r in rows if not r["deployable"]]
    counts = collections.Counter(r["genome"] for r in rows)
    repeated = [g for g, n in counts.items() if n > 1]
    # The waste is the re-evaluations, not the whole group: first sighting was work.
    seen: set[str] = set()
    redundant = []
    for r in rows:
        (redundant.append(r) if r["genome"] in seen else seen.add(r["genome"]))
    work = [(r["params"] + r["peak_act"]) / 1024 for r in ok]
    rank = sorted(ok, key=lambda r: -r["acc"]).index(best) + 1
    better = max((r for r in ok if r["acc"] > best["acc"]), key=lambda r: r["acc"])
    insert = [r for r in rows if r["mutation"].startswith("insert")]
    gen4 = {r["mutation"]: r for r in rows if r["gen"] == "4"}
    improved = {r["gen"] for r in ok if r["fitness"] > max(
        [x["fitness"] for x in ok[:ok.index(r)]] or [float("-inf")])} - {"0"}

    return [
        ("candidates evaluated", f"{len(rows)} candidates"),
        ("child slots", f"{len(rows) - 1} child slots"),
        ("rejected on budget", f"{len(rejected)} it threw away"),
        ("duplicate evaluations", f"{len(repeated)} it accidentally evaluated twice"),
        ("wall clock", f"{round(sum(r['train_s'] for r in rows) / 60)} minutes of training"),
        ("duplicate cost",
         f"{round(sum(r['train_s'] for r in redundant) / 60)} minutes of duplicated"),
        ("parameter cap", f"{PARAM_CAP:,} parameter cap"),
        ("sram cap", "250 KB"),
        ("training subset", "8,000 image subset"),
        ("subset fraction", "16% of the CIFAR-10"),

        ("seed accuracy", f"| seed, hand written | {seed['acc']:.4f} |"),
        ("seed params", f"| {seed['params']:,} |"),
        ("seed macs", f"| {seed['macs']:,} |"),
        ("seed peak", f"| {seed['peak_act']:,} |"),
        ("seed fitness", f"| {seed['fitness']:.4f} |"),
        ("best accuracy", f"**{best['acc']:.4f}**"),
        ("best params", f"| {best['params']:,} |"),
        ("best macs", f"**{best['macs']:,}**"),
        ("best fitness", f"**{best['fitness']:.4f}**"),

        ("accuracy gained", f"{100 * (best['acc'] - seed['acc']):.1f} accuracy points"),
        ("params added", f"{best['params'] - seed['params']} more\nparameters"),
        ("macs saved", f"{100 * (1 - best['macs'] / seed['macs']):.1f}% fewer"),
        ("where it was found",
         f"generation {best['gen']}, candidate {best['cand']}"),

        ("best working set", f"winner sits at {(best['params'] + best['peak_act']) / 1024:.1f} KB"),
        ("worst working set", f"was {max(work):.1f} KB"),
        ("sram headroom", f"{250 / max(work):.1f}x\nheadroom"),
        ("rank by accuracy", {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[rank] + " by accuracy"),
        ("accuracy that beat it", f"reached {better['acc']:.4f}"),
        ("macs of the model that beat it", f"{better['acc']:.4f} at {better['macs']:,}"),

        ("insert drawn", f"drawn {len(insert)} times and produced "
                         f"{sum(r['deployable'] for r in insert)} trainable"),
        ("gen 4 gain", f"| 0.5045 | +{gen4['kind block0 shift->asym']['acc'] - 0.4895:.4f} |"),
        ("gen 4 loss", f"| {gen4['kind block4 shift->asym']['acc']:.4f} | "
                       f"{gen4['kind block4 shift->asym']['acc'] - 0.4895:.4f} |"),
        ("generations that improved",
         ", ".join(sorted(improved)[:-1]) + " and " + sorted(improved)[-1]),
        ("generations that did not", f"{8 - len(improved)} of the 8 generations"),
    ]


def main() -> int:
    rows = load()
    readme = (ROOT / "README.md").read_text()

    best = max((r for r in rows if r["deployable"]), key=lambda r: r["fitness"])
    if json.loads(best["genome"]) != json.loads(
            (ROOT / "results" / "best_genome.json").read_text()):
        print("FAIL: best_genome.json is not the highest-fitness row in the log")
        return 1

    checked = claims(rows)
    missing = [(what, want) for what, want in checked if want not in readme]
    for what, want in missing:
        print(f"FAIL: README is missing {want!r}  ({what})")
    if missing:
        print(f"\n{len(missing)} of {len(checked)} checked figures are stale or missing.")
        return 1
    print(f"ok: {len(checked)} figures in README.md agree with results/search_log.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
