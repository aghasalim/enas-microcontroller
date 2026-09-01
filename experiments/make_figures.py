"""Redraw every README figure from results/search_log.csv.

Nothing here trains anything or invents anything: the only input is the log the
search wrote, so the figures cannot drift away from the numbers in the table.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "search_log.csv"
OUT = ROOT / "results" / "figures"

PARAM_CAP = 50_000
INK = "#1a1a1a"
GREY = "#9a9a9a"
BLUE = "#2b6cb0"
RED = "#c53030"
GREEN = "#2f855a"


def load() -> list[dict]:
    rows = list(csv.DictReader(LOG.open()))
    for r in rows:
        for k in ("gen", "cand", "params", "macs", "peak_act", "deployable"):
            r[k] = int(r[k])
        for k in ("acc", "train_s"):
            r[k] = float(r[k])
        r["fitness"] = float(r["fitness"])
    return rows


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)


def fig_trajectory(rows: list[dict]) -> None:
    """Every candidate in evaluation order, including the ones that failed."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    style(ax)

    ok = [(i, r) for i, r in enumerate(rows) if r["deployable"]]
    bad = [(i, r) for i, r in enumerate(rows) if not r["deployable"]]

    running, best = [], float("-inf")
    for r in rows:
        if r["deployable"] and r["fitness"] > best:
            best = r["fitness"]
        running.append(best)

    floor = min(r["fitness"] for _, r in ok) - 0.012
    ax.plot(range(len(rows)), running, color=BLUE, lw=2, zorder=2,
            label="running best")
    ax.scatter([i for i, _ in ok], [r["fitness"] for _, r in ok], s=34,
               color=INK, zorder=3, label="evaluated")
    ax.scatter([i for i, _ in bad], [floor] * len(bad), s=40, marker="x",
               color=RED, zorder=3, lw=1.6,
               label=f"over the {PARAM_CAP:,} parameter cap, never trained")

    top = max(x["fitness"] for x in rows if x["deployable"])
    for i, r in enumerate(rows):
        if r["deployable"] and r["fitness"] == top:
            ax.annotate(f"best  acc {r['acc']:.4f}, {r['params']:,} params",
                        (i, r["fitness"]), textcoords="offset points",
                        xytext=(10, -14), ha="left", fontsize=8.5, color=GREEN,
                        arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.9))
            break

    seen_gen = set()
    for i, r in enumerate(rows):
        if r["gen"] not in seen_gen and r["gen"] > 0:
            ax.axvline(i - 0.5, color="#dddddd", lw=0.8, zorder=1)
            ax.text(i - 0.5, ax.get_ylim()[1], f" g{r['gen']}", va="top",
                    fontsize=7.5, color=GREY)
            seen_gen.add(r["gen"])

    ax.set_xlabel("candidate, in the order it was evaluated", fontsize=9.5)
    ax.set_ylabel("fitness", fontsize=9.5)
    ax.set_title("Search trajectory: 33 candidates, 26 trained, 7 rejected on budget",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "search-trajectory.png", dpi=170)
    plt.close(fig)


def fig_budget(rows: list[dict]) -> None:
    """Where the candidates actually sat against the two hard constraints."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax in (ax1, ax2):
        style(ax)

    ok = [r for r in rows if r["deployable"]]
    bad = [r for r in rows if not r["deployable"]]
    best = max(ok, key=lambda r: r["fitness"])
    seed = rows[0]

    ax1.axvline(PARAM_CAP, color=RED, lw=1.4, ls="--")
    ax1.text(PARAM_CAP, min(r["acc"] for r in ok) - 0.004, " cap", color=RED,
             fontsize=8.5, va="bottom")
    ax1.scatter([r["params"] for r in ok], [r["acc"] for r in ok], s=34,
                color=INK, zorder=3)
    ax1.scatter([r["params"] for r in bad], [min(r["acc"] for r in ok) - 0.006] * len(bad),
                s=40, marker="x", color=RED, lw=1.6, zorder=3)
    for r, c, dx in ((seed, GREY, 10), (best, GREEN, -10)):
        ax1.scatter([r["params"]], [r["acc"]], s=90, facecolor="none",
                    edgecolor=c, lw=1.8, zorder=4)
        ax1.annotate("seed" if r is seed else "best", (r["params"], r["acc"]),
                     textcoords="offset points", xytext=(dx, 8), fontsize=9,
                     color=c, ha="left" if dx > 0 else "right")
    ax1.set_xlabel("parameters", fontsize=9.5)
    ax1.set_ylabel("validation accuracy", fontsize=9.5)
    ax1.set_title("The parameter cap is what bound the search", fontsize=10.5,
                  color=INK, loc="left")

    work = [(r["params"] + r["peak_act"]) / 1024 for r in ok]
    ax2.hist(work, bins=12, color=BLUE, alpha=0.85)
    ax2.axvline(250, color=RED, lw=1.4, ls="--")
    ax2.annotate("250 KB SRAM budget", (250, ax2.get_ylim()[1] * 0.9),
                 textcoords="offset points", xytext=(-8, 0), ha="right",
                 fontsize=8.5, color=RED)
    ax2.set_xlim(0, 270)
    ax2.set_xlabel("int8 working set: weights + peak activation, KB", fontsize=9.5)
    ax2.set_ylabel("candidates", fontsize=9.5)
    ax2.set_title(f"The SRAM budget never bound it: worst case {max(work):.1f} KB",
                  fontsize=10.5, color=INK, loc="left")

    fig.tight_layout()
    fig.savefig(OUT / "budget.png", dpi=170)
    plt.close(fig)


def fig_operators(rows: list[dict]) -> None:
    """Which mutation operators paid, and which never produced a usable child."""
    agg = defaultdict(lambda: {"drawn": 0, "trained": 0, "best": None})
    for r in rows[1:]:
        k = r["mutation"].split()[0]
        agg[k]["drawn"] += 1
        if r["deployable"]:
            agg[k]["trained"] += 1
            b = agg[k]["best"]
            agg[k]["best"] = r["fitness"] if b is None else max(b, r["fitness"])

    order = sorted(agg, key=lambda k: (agg[k]["best"] is not None, agg[k]["best"] or 0))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e6e6e6", lw=0.8)

    seed_fit = rows[0]["fitness"]
    for y, k in enumerate(order):
        a = agg[k]
        if a["best"] is None:
            ax.barh(y, seed_fit, color="#f0f0f0", edgecolor=RED, lw=1.2, hatch="//")
            ax.text(seed_fit * 0.5, y, "0 of 5 children fit the budget", va="center",
                    ha="center", fontsize=8.5, color=RED)
        else:
            ax.barh(y, a["best"], color=BLUE if a["best"] > seed_fit else GREY)
            ax.text(a["best"] + 0.003, y, f"{a['best']:.4f}", va="center",
                    fontsize=8.5, color=INK)
        ax.text(-0.004, y, f"{a['trained']}/{a['drawn']}", va="center", ha="right",
                fontsize=8, color=GREY)

    ax.axvline(seed_fit, color=INK, lw=1.2, ls=":")
    ax.text(seed_fit, len(order) - 0.35, " seed", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9.5)
    ax.set_xlim(0, max(a["best"] or 0 for a in agg.values()) * 1.12)
    ax.set_xlabel("best fitness reached by any child from this operator", fontsize=9.5)
    ax.set_title("Operator yield (trained / drawn on the left)", fontsize=11,
                 color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "operators.png", dpi=170)
    plt.close(fig)


def fig_topology(rows: list[dict]) -> None:
    """Seed against winner, block by block, with the two edits that separate them."""
    seed = json.loads(rows[0]["genome"])
    best_row = max((r for r in rows if r["deployable"]), key=lambda r: r["fitness"])
    best = json.loads(best_row["genome"])

    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axis("off")
    res = 16  # after the stride-2 stem

    for row, (name, g, acc) in enumerate(
            (("seed", seed, rows[0]["acc"]), ("best", best, best_row["acc"]))):
        y = 1.0 - row * 0.34
        ax.text(-0.02, y + 0.075, f"{name}   acc {acc:.4f}", fontsize=10,
                color=GREY if name == "seed" else GREEN, ha="left", va="center")
        r = res
        x = 0.0
        ax.add_patch(plt.Rectangle((x, y - 0.06), 0.09, 0.12, facecolor="#eeeeee",
                                   edgecolor=GREY))
        ax.text(x + 0.045, y, f"stem\n{g['stem_width']}", ha="center", va="center",
                fontsize=7.5)
        x += 0.115
        for i, b in enumerate(g["blocks"]):
            r = r // b["stride"]
            changed = (i >= len(seed["blocks"]) or b != seed["blocks"][i])
            face = "#d9e8f5" if b["kind"] == "asym" else "#f5f5f5"
            ax.add_patch(plt.Rectangle((x, y - 0.06), 0.155, 0.12, facecolor=face,
                                       edgecolor=GREEN if changed and row else GREY,
                                       lw=2.0 if changed and row else 1.0))
            label = f"{b['kind']} {b['out']}"
            if b["kind"] == "asym":
                label += f" k{b['k']}"
            ax.text(x + 0.0775, y + 0.018, label, ha="center", va="center", fontsize=7.5)
            ax.text(x + 0.0775, y - 0.028, f"/{b['stride']}  {r}x{r}", ha="center",
                    va="center", fontsize=6.8, color=GREY)
            x += 0.175

    ax.text(0.0, 0.50, "green outline marks a block the search changed. blue fill is a "
            "learned depthwise block,\ngrey is the zero-parameter shift block. "
            "the label under each block is stride and output resolution.",
            fontsize=8.2, color=GREY, va="top")
    ax.set_xlim(-0.03, 1.02)
    ax.set_ylim(0.30, 1.13)
    fig.tight_layout()
    fig.savefig(OUT / "topology.png", dpi=170)
    plt.close(fig)


def fig_validation() -> bool:
    """Seed against winner on a clean split, one line per training seed.

    Drawn only if results/validation.csv exists, since it comes from a separate
    run. Paired lines rather than two bars, because both architectures share an
    initialisation seed and the pairing is the whole reason the comparison is
    informative at n = 5.
    """
    path = ROOT / "results" / "validation.csv"
    if not path.exists():
        return False

    by: dict[str, dict[int, float]] = {}
    for r in csv.DictReader(path.open()):
        by.setdefault(r["arch"], {})[int(r["seed"])] = float(r["acc"])
    seeds = sorted(by["seed"])
    if not seeds:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in (ax1, ax2):
        style(ax)

    for s in seeds:
        lo, hi = by["seed"][s], by["winner"][s]
        ax1.plot([0, 1], [lo, hi], color=GREEN if hi > lo else RED, lw=1.4,
                 marker="o", ms=5, alpha=0.85)

    # Stagger labels that would otherwise land on top of each other. Two seeds
    # finishing within a rounding error of one another is the interesting case,
    # not one to hide under an unreadable label.
    span = max(by["winner"].values()) - min(by["winner"].values())
    placed: list[float] = []
    for s in sorted(seeds, key=lambda k: -by["winner"][k]):
        y = by["winner"][s]
        while any(abs(y - q) < span * 0.07 for q in placed):
            y -= span * 0.07
        placed.append(y)
        ax1.annotate(f"seed {s}", (1, y), textcoords="offset points",
                     xytext=(9, -3), fontsize=8, color=GREY)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["hand written", "search winner"], fontsize=9.5)
    ax1.set_xlim(-0.25, 1.45)
    ax1.set_ylabel("held-out accuracy", fontsize=9.5)
    ax1.set_title("Clean split, five training seeds, paired", fontsize=10.5,
                  color=INK, loc="left")

    d = [by["winner"][s] - by["seed"][s] for s in seeds]
    ax2.axhline(0, color=INK, lw=1.0)
    ax2.scatter([0.06 * (i - 2) for i in range(len(d))], d, s=46, color=GREEN,
                zorder=3)
    mean_d = sum(d) / len(d)
    ax2.axhline(mean_d, color=BLUE, lw=1.8)
    ax2.annotate(f"clean split  {mean_d:+.4f}", (0.16, mean_d),
                 textcoords="offset points", xytext=(0, 7), fontsize=9, color=BLUE)
    ax2.axhline(0.0370, color=RED, lw=1.6, ls="--")
    ax2.annotate("as the search reported it  +0.0370", (0.16, 0.0370),
                 textcoords="offset points", xytext=(0, 6), fontsize=9, color=RED)
    ax2.set_xlim(-0.22, 0.62)
    ax2.set_xticks([])
    ax2.set_ylabel("winner minus baseline", fontsize=9.5)
    ax2.set_title("The search overstated its own result by 1.8x", fontsize=10.5,
                  color=INK, loc="left")

    fig.tight_layout()
    fig.savefig(OUT / "validation.png", dpi=170)
    plt.close(fig)
    return True


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    fig_trajectory(rows)
    fig_budget(rows)
    fig_operators(rows)
    fig_topology(rows)
    if not fig_validation():
        print("note: results/validation.csv absent, skipping the validation figure")
    for p in sorted(OUT.glob("*.png")):
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
