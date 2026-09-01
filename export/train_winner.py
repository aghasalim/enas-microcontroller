"""Train the winning genome on a clean split and save weights for export.

The search never saved a checkpoint, only genomes, so there is nothing to
deploy until this runs. Validation is carved out of the training split here,
not taken from the test split, so the accuracy this prints is an honest
held-out number rather than the selection-biased one the search reported.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.validate_winner import clean_loaders  # noqa: E402
from search import fitness as FIT                      # noqa: E402
from search.space import build                         # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--train-n", type=int, default=40_000)
    p.add_argument("--val-n", type=int, default=5_000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    genome = json.loads((ROOT / "results" / "best_genome.json").read_text())
    tl, vl = clean_loaders(a.train_n, a.val_n, a.batch)

    torch.manual_seed(a.seed)
    model = build(genome)
    n = FIT.count_params(model)
    print(f"training the winner: {n:,} parameters, {a.epochs} epochs on "
          f"{a.train_n:,} images", flush=True)

    t0 = time.perf_counter()
    acc, _ = FIT.train_micro(model, tl, vl, a.epochs)
    secs = time.perf_counter() - t0

    out = ROOT / "results" / "winner.pt"
    torch.save({"genome": genome, "state_dict": model.state_dict(),
                "acc": round(acc, 4), "epochs": a.epochs, "train_n": a.train_n,
                "val_n": a.val_n, "seed": a.seed}, out)
    print(f"\nheld-out accuracy {acc:.4f} on {a.val_n:,} images "
          f"never seen in training, {secs / 60:.0f} min")
    print(f"-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
