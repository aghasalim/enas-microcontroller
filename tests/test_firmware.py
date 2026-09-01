"""The C implementation is checked from pytest too, so one command covers both.

These shell out rather than reimplement anything. The equivalence test is the C
program in firmware/, and this only builds it and reports what it said.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FW = ROOT / "firmware"
GEN = FW / "generated"

needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")


def test_export_is_present_and_consistent():
    """The generated header and the binaries must describe the same network."""
    header = (GEN / "micronet_arch.h").read_text()
    n = {k: int(header.split(f"#define MICRONET_{k}")[1].split()[0])
         for k in ("N_WEIGHTS", "N_INDICES", "N_GOLDEN", "CLASSES", "PEAK")}
    assert (GEN / "micronet_weights.bin").stat().st_size == n["N_WEIGHTS"] * 4
    assert (GEN / "micronet_indices.bin").stat().st_size == n["N_INDICES"] * 4
    assert (GEN / "micronet_golden.bin").stat().st_size == \
        n["N_GOLDEN"] * (3 * 32 * 32 + n["CLASSES"]) * 4


def test_exported_peak_matches_the_search_log():
    """The C header's peak activation is the number the search optimised."""
    import csv
    import json

    header = (GEN / "micronet_arch.h").read_text()
    peak = int(header.split("#define MICRONET_PEAK")[1].split()[0])
    best_genome = json.loads((ROOT / "results" / "best_genome.json").read_text())
    rows = [r for r in csv.DictReader((ROOT / "results" / "search_log.csv").open())
            if json.loads(r["genome"]) == best_genome]
    assert rows, "best_genome.json does not appear in the log"
    assert peak == int(rows[0]["peak_act"])


def test_c_op_table_agrees_with_pytorch_on_macs():
    """Two independent MAC counts, one from the torch graph and one from the
    exported op table, over the same network."""
    import csv
    import json

    header = (GEN / "micronet_arch.h").read_text()
    body = header.split("MICRONET_OPS[MICRONET_N_OPS] = {")[1]
    fields = ["op", "in_c", "in_h", "in_w", "out_c", "out_h", "out_w", "kh", "kw",
              "sh", "sw", "ph", "pw", "groups", "w_off", "b_off", "extra"]
    total = 0
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        vals = [int(v) for v in line[1:line.index("}")].split(",")]
        o = dict(zip(fields, vals))
        if o["op"] == 0:        # OP_CONV
            total += o["out_c"] * o["out_h"] * o["out_w"] \
                   * (o["in_c"] // o["groups"]) * o["kh"] * o["kw"]
        elif o["op"] == 6:      # OP_LINEAR
            total += o["in_c"] * o["out_c"]

    best = json.loads((ROOT / "results" / "best_genome.json").read_text())
    logged = [int(r["macs"]) for r in
              csv.DictReader((ROOT / "results" / "search_log.csv").open())
              if json.loads(r["genome"]) == best]
    assert total == logged[0]


@needs_cc
def test_c_forward_matches_pytorch():
    subprocess.run(["make", "-C", str(FW), "clean"], check=True,
                   capture_output=True)
    r = subprocess.run(["make", "-C", str(FW), "test"], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "reproduces PyTorch on every golden image" in r.stdout, r.stdout


@needs_cc
def test_equivalence_test_would_catch_a_broken_kernel():
    """A test that cannot fail is not evidence. Break the residual add, confirm
    the C test rejects it, and put the file back."""
    src = FW / "micronet.c"
    original = src.read_text()
    broken = original.replace("dst[oy * o->out_w + ox] += v;",
                              "dst[oy * o->out_w + ox] = v;")
    assert broken != original, "the line this test perturbs has moved"
    try:
        src.write_text(broken)
        subprocess.run(["make", "-C", str(FW), "clean"], check=True,
                       capture_output=True)
        r = subprocess.run(["make", "-C", str(FW), "test"], capture_output=True,
                           text=True)
        assert r.returncode != 0, "the C test passed on a deliberately broken kernel"
    finally:
        src.write_text(original)
        subprocess.run(["make", "-C", str(FW), "clean"], check=True,
                       capture_output=True)
