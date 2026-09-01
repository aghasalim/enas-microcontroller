"""Checks that tie the code to the published log.

The point of most of these is not that the search is clever, it is that
results/search_log.csv still describes the code in this repository. If someone
edits a block definition, the logged parameter counts stop matching and the
suite fails rather than the README quietly becoming fiction.
"""
from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search import fitness as FIT  # noqa: E402
from search.space import build, mutate, seed_genome  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHAPE = (3, 32, 32)


@pytest.fixture(scope="module")
def log() -> list[dict]:
    return list(csv.DictReader((ROOT / "results" / "search_log.csv").open()))


# --- the fitness caps are rejections, not penalties ---------------------------

def test_over_param_cap_is_rejected():
    assert FIT.fitness(0.99, 50_001, 1, 1) == float("-inf")
    assert FIT.fitness(0.99, 50_000, 1, 1) > float("-inf")


def test_over_sram_is_rejected():
    """A model inside the parameter cap can still fail on activations."""
    cap = 250 * 1024
    assert FIT.fitness(0.99, 40_000, 1, cap - 40_000 + 1) == float("-inf")
    assert FIT.fitness(0.99, 40_000, 1, cap - 40_000) > float("-inf")


def test_a_perfect_undeployable_model_loses_to_a_poor_deployable_one():
    """The whole reason the caps are hard: accuracy must not buy its way out."""
    assert FIT.fitness(1.00, 60_000, 1_000, 1_000) < FIT.fitness(0.10, 10_000, 1_000, 1_000)


def test_fitness_penalises_size_and_compute():
    base = FIT.fitness(0.5, 20_000, 2_000_000, 8_000)
    assert FIT.fitness(0.5, 40_000, 2_000_000, 8_000) < base
    assert FIT.fitness(0.5, 20_000, 4_000_000, 8_000) < base


# --- the mutation operator ----------------------------------------------------

def test_mutation_never_returns_the_parent():
    """A child equal to its parent spends a full training run on a known point."""
    rng = random.Random(0)
    g = seed_genome()
    for _ in range(200):
        child, how = mutate(g, rng)
        assert child != g, how


def test_kernel_edits_only_target_learned_blocks():
    """Shift blocks have no spatial weights, so a kernel edit on one is a no-op."""
    rng = random.Random(1)
    g = seed_genome()
    for _ in range(400):
        child, how = mutate(g, rng)
        if how.startswith("kernel"):
            i = int(how.split()[1].removeprefix("block"))
            assert g["blocks"][i]["kind"] == "asym", how


def test_one_edit_per_child():
    """Fitness changes are only attributable if a child differs in one place."""
    rng = random.Random(2)
    g = seed_genome()
    for _ in range(150):
        child, how = mutate(g, rng)
        if how.startswith(("insert", "drop")):
            continue                      # these shift every later index by one
        diff = sum(b != c for b, c in zip(g["blocks"], child["blocks"]))
        diff += g["act_shift"] != child["act_shift"]
        assert diff == 1, f"{how} changed {diff} things"


# --- the model itself ---------------------------------------------------------

def test_build_is_deterministic_under_a_seed():
    """Seeding before construction is what puts the initial weights in the seed."""
    torch.manual_seed(7)
    a = build(seed_genome())
    torch.manual_seed(7)
    b = build(seed_genome())
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p, q)


def test_peak_activation_is_the_largest_tensor_not_the_last():
    """Hand count for the seed: the stride-2 stem gives 24x16x16, and the first
    block widens that to 48x16x16 = 12,288, which is the peak. Everything after
    it is smaller, so a metric that read the output shape would report 10."""
    assert FIT.peak_activation(build(seed_genome()), SHAPE) == 48 * 16 * 16


def test_asym_block_is_cheaper_than_the_square_kernel_it_replaces():
    from search.blocks import AsymBlock

    dw = [m for m in AsymBlock(32, 32, k=5).modules()
          if isinstance(m, torch.nn.Conv2d) and m.groups == 32]
    assert sum(m.weight.numel() for m in dw) == 32 * 10        # 1x5 plus 5x1
    assert 32 * 10 < 32 * 25                                    # against 5x5


# Pinned from the code as published. Shape and cost tests cannot see a change to
# the forward pass: deleting a residual connection leaves every tensor the same
# size, the same parameter count and the same MAC count, so it slips past all of
# them. This pins the arithmetic itself. If the model is meant to change, this
# value is meant to be regenerated, deliberately and in the same commit.
SEED_FORWARD = [
    [-0.194556, -0.178867, 0.430614, -0.121499, 0.288177,
     0.133701, -0.110003, -0.495626, -0.031319, 0.404465],
    [-0.199993, -0.175291, 0.406320, -0.125422, 0.266369,
     0.137817, -0.111007, -0.448011, 0.001585, 0.398713],
]


def test_seed_forward_pass_is_unchanged():
    torch.manual_seed(0)
    model = build(seed_genome()).eval()
    torch.manual_seed(1)
    with torch.no_grad():
        y = model(torch.randn(2, *SHAPE))
    assert y.shape == (2, 10)
    torch.testing.assert_close(y, torch.tensor(SEED_FORWARD), rtol=1e-4, atol=1e-4)


def test_shift_block_spends_no_weights_on_spatial_mixing():
    from search.blocks import GroupShift

    assert sum(p.numel() for p in GroupShift(40).parameters()) == 0


# --- the log still describes this code ---------------------------------------

def test_every_logged_genome_still_builds_to_its_logged_cost(log):
    """33 independent ties between the code and the published numbers."""
    for r in log:
        g = json.loads(r["genome"])
        torch.manual_seed(0)
        m = build(g)
        assert FIT.count_params(m) == int(r["params"]), r["mutation"]
        assert FIT.macs(m, SHAPE) == int(r["macs"]), r["mutation"]
        assert FIT.peak_activation(m, SHAPE) == int(r["peak_act"]), r["mutation"]


def test_logged_fitness_is_what_the_fitness_function_returns(log):
    for r in log:
        want = FIT.fitness(float(r["acc"]), int(r["params"]), int(r["macs"]),
                           int(r["peak_act"]))
        got = float(r["fitness"])
        assert (want == got == float("-inf")) or math.isclose(want, got, rel_tol=1e-12)


def test_rejected_candidates_were_never_trained(log):
    """Budget is checked before training, so a rejection must cost no time."""
    for r in log:
        if r["deployable"] == "0":
            assert float(r["train_s"]) == 0.0
            assert float(r["acc"]) == 0.0
            assert int(r["params"]) > 50_000


def test_best_genome_json_is_the_highest_fitness_row(log):
    best = max((r for r in log if r["deployable"] == "1"),
               key=lambda r: float(r["fitness"]))
    stored = json.loads((ROOT / "results" / "best_genome.json").read_text())
    assert json.loads(best["genome"]) == stored


def test_mutation_labels_match_the_genomes_they_produced(log):
    """Replay the best-so-far chain the controller followed and check that each
    label describes the edit that genome actually contains."""
    parent = json.loads(log[0]["genome"])
    best_fit = float(log[0]["fitness"])
    for r in log[1:]:
        child, how = json.loads(r["genome"]), r["mutation"]
        if how.startswith(("width", "kernel", "kind", "stride")):
            i = int(how.split()[1].removeprefix("block"))
            changed = [j for j in range(len(parent["blocks"]))
                       if parent["blocks"][j] != child["blocks"][j]]
            assert changed == [i], f"{how} really changed blocks {changed}"
        elif how.startswith("act"):
            assert parent["blocks"] == child["blocks"]
            assert parent["act_shift"] != child["act_shift"], how
        elif how.startswith("drop"):
            assert len(child["blocks"]) == len(parent["blocks"]) - 1, how
        elif how.startswith("insert"):
            assert len(child["blocks"]) == len(parent["blocks"]) + 1, how
        if r["deployable"] == "1" and float(r["fitness"]) > best_fit:
            parent, best_fit = child, float(r["fitness"])


def test_readme_numbers_match_the_log():
    r = subprocess.run([sys.executable, str(ROOT / "check_numbers.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
