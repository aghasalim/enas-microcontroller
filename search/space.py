"""The architecture space, and the mutations that move through it.

A genome is a small dict, not code. That is deliberate: a mutation is then a
edit to a dict that can be logged, diffed and replayed, and every architecture
in the run is reconstructible from its genome alone. Generating code per
candidate would make the run unreproducible for the sake of looking clever.
"""
from __future__ import annotations

import copy
import random

import torch.nn as nn

from .blocks import AsymBlock, MicroNetGen1, PowerOfTwoReLU, ShiftBlock  # noqa: F401

BLOCK_KINDS = ("shift", "asym")
WIDTH_CHOICES = (16, 24, 32, 48, 64, 96, 128, 160)
KERNEL_CHOICES = (3, 5, 7)
SHIFT_CHOICES = (2, 3, 4, 5)


def seed_genome() -> dict:
    """Gen-0. Deliberately the hand-written baseline, so the search has to beat
    something a person would actually have written rather than a straw man."""
    return {
        "stem_width": 24,
        "act_shift": 4,
        "blocks": [
            {"kind": "shift", "out": 48, "stride": 1, "k": 3},
            {"kind": "asym", "out": 96, "stride": 2, "k": 5},
            {"kind": "shift", "out": 96, "stride": 1, "k": 3},
            {"kind": "asym", "out": 128, "stride": 2, "k": 5},
            {"kind": "shift", "out": 128, "stride": 1, "k": 3},
        ],
    }


def mutate(genome: dict, rng: random.Random, tries: int = 12) -> tuple[dict, str]:
    """One edit per child, so a fitness change is attributable to one cause.

    Retries if the edit produced the parent again, which otherwise spends a full
    training run confirming something already known.
    """
    for _ in range(tries):
        child, how = _mutate_once(genome, rng)
        if child != genome:
            return child, how
    return _mutate_once(genome, rng)


def _mutate_once(genome: dict, rng: random.Random) -> tuple[dict, str]:
    g = copy.deepcopy(genome)
    ops = ["width", "kernel", "kind", "act", "depth_add", "depth_drop", "stride"]
    if len(g["blocks"]) <= 3:
        ops.remove("depth_drop")
    if len(g["blocks"]) >= 7:
        ops.remove("depth_add")
    # A kernel edit on a shift block changes nothing, because shift blocks have
    # no spatial weights. Offering it wastes a training slot on a duplicate.
    if not any(b["kind"] == "asym" for b in g["blocks"]):
        ops.remove("kernel")
    op = rng.choice(ops)
    if op == "kernel":
        i = rng.choice([j for j, b in enumerate(g["blocks"]) if b["kind"] == "asym"])
    else:
        i = rng.randrange(len(g["blocks"]))

    if op == "width":
        cur = g["blocks"][i]["out"]
        choices = [w for w in WIDTH_CHOICES if w != cur]
        g["blocks"][i]["out"] = rng.choice(choices)
        return g, f"width block{i} {cur}->{g['blocks'][i]['out']}"
    if op == "kernel":
        cur = g["blocks"][i]["k"]
        g["blocks"][i]["k"] = rng.choice([k for k in KERNEL_CHOICES if k != cur])
        return g, f"kernel block{i} {cur}->{g['blocks'][i]['k']}"
    if op == "kind":
        cur = g["blocks"][i]["kind"]
        g["blocks"][i]["kind"] = "asym" if cur == "shift" else "shift"
        return g, f"kind block{i} {cur}->{g['blocks'][i]['kind']}"
    if op == "act":
        cur = g["act_shift"]
        g["act_shift"] = rng.choice([s for s in SHIFT_CHOICES if s != cur])
        return g, f"act leak 2^-{cur}->2^-{g['act_shift']}"
    if op == "depth_add":
        prev = g["blocks"][i]["out"]
        g["blocks"].insert(i + 1, {"kind": rng.choice(BLOCK_KINDS), "out": prev,
                                   "stride": 1, "k": rng.choice(KERNEL_CHOICES)})
        return g, f"insert block after {i}"
    if op == "depth_drop":
        g["blocks"].pop(i)
        return g, f"drop block{i}"
    # stride: keep at most three downsamples or the map vanishes
    cur = g["blocks"][i]["stride"]
    new = 1 if cur == 2 else 2
    if new == 2 and sum(b["stride"] for b in g["blocks"]) >= 8:
        return mutate(genome, rng)
    g["blocks"][i]["stride"] = new
    return g, f"stride block{i} {cur}->{new}"


def build(genome: dict, num_classes: int = 10, in_ch: int = 3) -> nn.Module:
    """Genome to module. The only place a genome becomes weights."""
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            w = genome["stem_width"]
            s = genome["act_shift"]
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, w, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(w), PowerOfTwoReLU(s))
            layers, prev = [], w
            for b in genome["blocks"]:
                cls = ShiftBlock if b["kind"] == "shift" else AsymBlock
                kw = {} if b["kind"] == "shift" else {"k": b["k"]}
                layers.append(cls(prev, b["out"], stride=b["stride"], shift=s, **kw))
                prev = b["out"]
            self.blocks = nn.Sequential(*layers)
            self.head = nn.Linear(prev, num_classes)

        def forward(self, x):
            x = self.blocks(self.stem(x))
            return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))

    return Net()
