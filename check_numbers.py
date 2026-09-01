"""Fail if any figure quoted in README.md disagrees with the files it came from.

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
import math
import statistics as st
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


def binom_sd(ok: list[dict], n: int = 2000) -> float:
    """Standard deviation of an accuracy estimate from n validation samples."""
    p = st.mean(r["acc"] for r in ok)
    return math.sqrt(p * (1 - p) / n)


def pooled_se(a: dict, b: dict, n: int = 2000) -> float:
    return math.sqrt(sum(r["acc"] * (1 - r["acc"]) / n for r in (a, b)))


def arch() -> dict[str, int]:
    """The MICRONET_* defines from the generated header."""
    text = (ROOT / "firmware" / "generated" / "micronet_arch.h").read_text()
    return {k: int(text.split(f"#define MICRONET_{k}")[1].split()[0])
            for k in ("N_OPS", "N_WEIGHTS", "PEAK", "CLASSES")}


def ref_working_set_kb() -> float:
    """What firmware/micronet.c actually needs: folded weights plus the three
    full size buffers it ping-pongs between, one byte each at int8."""
    a = arch()
    return (a["N_WEIGHTS"] + 3 * a["PEAK"]) / 1024


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

    # Anchors carry surrounding words only where the value is short enough to
    # match by accident. Distinctive values stand alone, because prose rewraps
    # and an anchor that spans a line break fails on a reflow rather than on a
    # real disagreement.
    return [
        ("candidates evaluated", f"{len(rows)} candidates"),
        ("child slots", f"{len(rows) - 1} child slots"),
        ("rejected on budget", f"{len(rejected)} rejected"),
        ("duplicate evaluations", f"{len(repeated)} evaluated twice"),
        ("wall clock", f"{round(sum(r['train_s'] for r in rows) / 60)} minutes of training"),
        ("duplicate cost",
         f"{round(sum(r['train_s'] for r in redundant) / 60)} minutes of duplicated"),
        ("parameter cap", f"{PARAM_CAP:,} cap"),
        ("parameter cap in the objective", "50{,}000"),
        ("sram cap", "250 KB"),
        ("training subset", "8,000 training images"),
        ("subset fraction", "16% of the split"),
        ("validation size", "2,000 validation"),

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
        ("params added", f"{best['params'] - seed['params']} more"),
        ("macs saved", f"{100 * (1 - best['macs'] / seed['macs']):.1f}% fewer"),
        ("where it was found",
         f"generation {best['gen']}, candidate {best['cand']}"),

        ("best working set", f"{(best['params'] + best['peak_act']) / 1024:.1f} KB"),
        ("worst working set", f"{max(work):.1f} KB"),
        ("sram headroom", f"{250 / max(work):.1f}x"),
        ("rank by accuracy", {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[rank] + " by accuracy"),
        ("the model that beat it", f"{better['acc']:.4f} at {better['macs']:,}"),

        ("insert drawn", f"drawn {len(insert)} times and produced "
                         f"{sum(r['deployable'] for r in insert)} trainable"),
        ("gen 4 gain", f"| 0.5045 | +{gen4['kind block0 shift->asym']['acc'] - 0.4895:.4f} |"),
        ("gen 4 loss", f"| {gen4['kind block4 shift->asym']['acc']:.4f} | "
                       f"{gen4['kind block4 shift->asym']['acc'] - 0.4895:.4f} |"),
        ("generations that improved",
         ", ".join(sorted(improved)[:-1]) + " and " + sorted(improved)[-1]),
        ("generations that did not", f"{8 - len(improved)} of the 8 generations"),

        # Section 6. These are the honesty of the report, so they are checked
        # like any other figure rather than trusted as prose.
        ("trained candidates", f"{len(ok)} trained candidates"),
        ("observed sd", f"{st.stdev([r['acc'] for r in ok]):.4f}"),
        ("binomial sd", f"{binom_sd(ok):.4f}"),
        ("noise share of variance", f"{100 * binom_sd(ok) ** 2 / st.stdev([r['acc'] for r in ok]) ** 2:.0f}%"),
        ("residual sd",
         f"{(st.stdev([r['acc'] for r in ok]) ** 2 - binom_sd(ok) ** 2) ** 0.5:.4f}"),
        ("headline gap", f"{best['acc'] - seed['acc']:.4f}"),
        ("pooled standard error", f"{pooled_se(seed, best):.4f}"),
        ("z score", f"z = {(best['acc'] - seed['acc']) / pooled_se(seed, best):.2f}"),

        # Section 7. Derived from the generated op table, so a re-export that
        # changes the network invalidates these the same way it invalidates
        # the header.
        ("exported ops", f"{arch()['N_OPS']} ops"),
        ("reference working set", f"{ref_working_set_kb():.1f} KB at int8"),
        ("cost model gap",
         f"{ref_working_set_kb() / ((best['params'] + best['peak_act']) / 1024):.1f}x"),
    ]


def validation_claims() -> list[tuple[str, str]]:
    """Figures from the clean-split retest. Empty if it has not been run."""
    path = ROOT / "results" / "validation.csv"
    if not path.exists():
        return []

    by: dict[str, dict[int, float]] = {}
    for r in csv.DictReader(path.open()):
        by.setdefault(r["arch"], {})[int(r["seed"])] = float(r["acc"])
    seeds = sorted(by.get("seed", {}))
    if not seeds or len(seeds) < 2:
        return []

    d = [by["winner"][k] - by["seed"][k] for k in seeds]
    mean_d = st.mean(d)
    se = st.stdev(d) / math.sqrt(len(d))
    search_gap = 0.0370

    return [
        ("retest seeds", f"{len(seeds)} training seeds"),
        ("retest baseline mean", f"| baseline | {st.mean(by['seed'].values()):.4f} |"),
        ("retest baseline sd", f"| {st.stdev(by['seed'].values()):.4f} |"),
        ("retest winner mean", f"| winner | {st.mean(by['winner'].values()):.4f} |"),
        ("retest winner sd", f"| {st.stdev(by['winner'].values()):.4f} |"),
        ("paired differences",
         "| " + " | ".join(f"{x:+.4f}" for x in d) + " |"),
        ("retest mean gap", f"Mean {mean_d:+.4f}"),
        ("retest standard error", f"standard error {se:.4f}"),
        ("retest t", f"t(4) = {mean_d / se:.2f}"),
        ("retest wins", f"ahead on {sum(x > 0 for x in d)} of {len(d)} seeds"),
        ("inflation factor", f"{search_gap / mean_d:.1f}x"),
        ("gap in points", f"**{100 * mean_d:.1f} points**"),
        ("seed spread against the gap",
         f"{st.stdev(by['winner'].values()) / mean_d * 100:.0f}% of the "
         f"{mean_d:.4f} effect"),
        ("weakest seed", f"the gap is {min(d):+.4f}"),
    ]


def main() -> int:
    rows = load()
    readme = (ROOT / "README.md").read_text()

    best = max((r for r in rows if r["deployable"]), key=lambda r: r["fitness"])
    if json.loads(best["genome"]) != json.loads(
            (ROOT / "results" / "best_genome.json").read_text()):
        print("FAIL: best_genome.json is not the highest-fitness row in the log")
        return 1

    checked = claims(rows) + validation_claims()
    missing = [(what, want) for what, want in checked if want not in readme]
    for what, want in missing:
        print(f"FAIL: README is missing {want!r}  ({what})")
    if missing:
        print(f"\n{len(missing)} of {len(checked)} checked figures are stale or missing.")
        return 1
    print(f"ok: {len(checked)} figures in README.md agree with the search log, "
          f"the clean-split retest and the generated op table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
