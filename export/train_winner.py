"""Train the winning genome on a clean split and save weights for export.

The search never saved a checkpoint, only genomes, so there is nothing to
deploy until this runs. Validation is carved out of the training split here,
not taken from the test split, so the accuracy this prints is an honest
held-out number rather than the selection-biased one the search reported.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.validate_winner import clean_loaders  # noqa: E402
from search import fitness as FIT                      # noqa: E402
from search.space import build, seed_genome            # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--train-n", type=int, default=40_000)
    p.add_argument("--val-n", type=int, default=5_000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--genome", default="results/best_genome.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/winner.pt")
    p.add_argument("--log", help="append arch, seed, accuracy and time to this CSV")
    p.add_argument("--name", default="winner", help="arch name written to --log")
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    # "seed" trains the hand-written baseline, for a like-for-like comparison.
    genome = (seed_genome() if a.genome == "seed"
              else json.loads((ROOT / a.genome).read_text()))
    tl, vl = clean_loaders(a.train_n, a.val_n, a.batch)

    torch.manual_seed(a.seed)
    model = build(genome)
    n = FIT.count_params(model)
    print(f"training the winner: {n:,} parameters, {a.epochs} epochs on "
          f"{a.train_n:,} images", flush=True)

    t0 = time.perf_counter()
    acc, _ = FIT.train_micro(model, tl, vl, a.epochs, device=a.device)
    model.to("cpu")
    secs = time.perf_counter() - t0

    out = ROOT / a.out
    torch.save({"genome": genome, "state_dict": model.state_dict(),
                "acc": round(acc, 4), "epochs": a.epochs, "train_n": a.train_n,
                "val_n": a.val_n, "seed": a.seed}, out)
    print(f"\nheld-out accuracy {acc:.4f} on {a.val_n:,} images "
          f"never seen in training, {secs / 60:.0f} min")
    print(f"-> {out.relative_to(ROOT)}")
    if a.log:
        path = ROOT / a.log
        new = not path.exists()
        with path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["arch", "seed", "acc", "params", "epochs", "train_n", "train_s"])
            w.writerow([a.name, a.seed, round(acc, 4), n, a.epochs, a.train_n, round(secs, 1)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
