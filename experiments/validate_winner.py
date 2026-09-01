"""Retest seed against winner on a clean split, over several training seeds.

The search itself has two defects this script exists to measure. It ranked
candidates on a 2,000 image subset of the official CIFAR-10 test split, which is
model selection on test, and it trained every candidate from one initialisation,
which confounds architecture quality with initialisation luck.

Here the validation set is held out of the training split instead, the test split
is never touched, and each architecture is trained from several seeds so the gap
can be compared against its own spread. Everything else, the optimiser, the
schedule, the augmentation, the epoch count and the 8,000 image training budget,
is exactly what the search used, so the only things that change are the split and
the seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from search import fitness as FIT          # noqa: E402
from search.space import build, seed_genome  # noqa: E402

RESULTS = ROOT / "results"
FIELDS = ["arch", "seed", "acc", "params", "train_s"]


def clean_loaders(train_n: int, val_n: int, batch: int, split_seed: int = 1234):
    """Validation is carved out of the training split. The test split is not
    opened, here or anywhere else in this repository."""
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])
    tf_eval = transforms.Compose([transforms.ToTensor(), norm])
    tr = datasets.CIFAR10(ROOT / "data", train=True, transform=tf_train)
    va = datasets.CIFAR10(ROOT / "data", train=True, transform=tf_eval)
    g = torch.Generator().manual_seed(split_seed)
    order = torch.randperm(len(tr), generator=g).tolist()
    val_idx, train_idx = order[:val_n], order[val_n:val_n + train_n]
    assert not set(val_idx) & set(train_idx)
    return (DataLoader(Subset(tr, train_idx), batch_size=batch, shuffle=True, num_workers=2),
            DataLoader(Subset(va, val_idx), batch_size=256, num_workers=2))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-n", type=int, default=8000)
    p.add_argument("--val-n", type=int, default=5000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    tl, vl = clean_loaders(a.train_n, a.val_n, a.batch)
    arches = {
        "seed": seed_genome(),
        "winner": json.loads((RESULTS / "best_genome.json").read_text()),
    }

    out = RESULTS / "validation.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        got: dict[str, list[float]] = {k: [] for k in arches}
        for s in range(a.seeds):
            for name, genome in arches.items():
                torch.manual_seed(s)              # before build, as in the search
                model = build(genome)
                t0 = time.perf_counter()
                acc, _ = FIT.train_micro(model, tl, vl, a.epochs)
                w.writerow({"arch": name, "seed": s, "acc": round(acc, 4),
                            "params": FIT.count_params(model),
                            "train_s": round(time.perf_counter() - t0, 2)})
                fh.flush()
                got[name].append(acc)
                print(f"seed {s}  {name:6s}  acc {acc:.4f}  "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    print()
    for name, xs in got.items():
        print(f"{name:6s}  mean {st.mean(xs):.4f}  sd {st.stdev(xs):.4f}  "
              f"min {min(xs):.4f}  max {max(xs):.4f}")
    gap = st.mean(got["winner"]) - st.mean(got["seed"])
    pooled = (st.stdev(got["winner"]) ** 2 + st.stdev(got["seed"]) ** 2) ** 0.5
    wins = sum(w > s for w, s in zip(got["winner"], got["seed"]))
    print(f"\ngap {gap:+.4f}  spread of the two means {pooled:.4f}  "
          f"winner ahead on {wins} of {a.seeds} paired seeds")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
