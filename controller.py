"""Evolutionary architecture search under a hard microcontroller budget.

Every candidate ever evaluated is written to results/search_log.csv, including
the ones that got worse and the ones that were rejected as undeployable. A
search that only reports its winner is not reproducible and not honest about
how much of the result was luck.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from search import fitness as FIT
from search.space import build, mutate, random_genome, seed_genome

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SHAPE = (3, 32, 32)

FIELDS = ["gen", "cand", "parent", "mutation", "params", "macs", "peak_act",
          "acc", "fitness", "deployable", "train_s", "genome", "accs"]


def loaders(train_n: int, val_n: int, batch: int, seed: int,
            val_from_train: bool = False) -> tuple[DataLoader, DataLoader]:
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])
    tf_eval = transforms.Compose([transforms.ToTensor(), norm])
    tr = datasets.CIFAR10(ROOT / "data", train=True, transform=tf_train, download=True)
    g = torch.Generator().manual_seed(seed)
    if val_from_train:
        # Validation carved out of the training split, so the search never
        # selects on the test split. Same carve as experiments/validate_winner.py.
        va = datasets.CIFAR10(ROOT / "data", train=True, transform=tf_eval, download=True)
        order = torch.randperm(len(tr), generator=g).tolist()
        va_idx, tr_idx = order[:val_n], order[val_n:val_n + train_n]
    else:
        va = datasets.CIFAR10(ROOT / "data", train=False, transform=tf_eval, download=True)
        tr_idx = torch.randperm(len(tr), generator=g)[:train_n].tolist()
        va_idx = torch.randperm(len(va), generator=g)[:val_n].tolist()
    return (DataLoader(Subset(tr, tr_idx), batch_size=batch, shuffle=True, num_workers=2),
            DataLoader(Subset(va, va_idx), batch_size=256, num_workers=2))


def assess(genome: dict, tl, vl, epochs: int, seed: int, seeds: int = 1,
           device: str = "cpu") -> dict:
    # Seed before the model is built. Constructing first and seeding after leaves
    # the initial weights outside the seed, which makes a run unreproducible in a
    # way that is invisible until two machines disagree.
    torch.manual_seed(seed)
    model = build(genome)
    params = FIT.count_params(model)
    peak = FIT.peak_activation(model, SHAPE)
    mac = FIT.macs(model, SHAPE)
    f_pre = FIT.fitness(0.0, params, mac, peak)
    if f_pre == float("-inf"):          # undeployable: do not spend training on it
        return {"params": params, "macs": mac, "peak_act": peak, "acc": 0.0,
                "fitness": float("-inf"), "deployable": 0, "train_s": 0.0, "accs": "[]"}
    # With seeds > 1 the candidate is ranked on its mean over several
    # initialisations, so a single lucky draw cannot win a generation.
    accs, secs = [], 0.0
    for s in range(seeds):
        torch.manual_seed(seed + s)
        m = model if s == 0 else build(genome)
        acc, t = FIT.train_micro(m, tl, vl, epochs, device=device)
        accs.append(acc)
        secs += t
    acc = sum(accs) / len(accs)
    return {"params": params, "macs": mac, "peak_act": peak, "acc": acc,
            "fitness": FIT.fitness(acc, params, mac, peak), "deployable": 1,
            "train_s": secs, "accs": json.dumps(accs)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--children", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-n", type=int, default=8000)
    p.add_argument("--val-n", type=int, default=2000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seeds", type=int, default=1,
                   help="training seeds per candidate, ranked on the mean")
    p.add_argument("--device", default="cpu")
    p.add_argument("--val-from-train", action="store_true",
                   help="hold validation out of the training split, never the test split")
    p.add_argument("--random", type=int, default=0, metavar="N",
                   help="random search baseline: N uniform draws instead of the hill climb")
    p.add_argument("--out", default="search_log.csv", help="file name under results/")
    a = p.parse_args()

    RESULTS.mkdir(exist_ok=True)
    # Timed on this machine before the run: 4 threads beat 6. These models are
    # small enough that thread overhead and the efficiency cores cost more than
    # the extra parallelism returns. The train_s column of search_log.csv is the
    # authority on how long the run itself took.
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tl, vl = loaders(a.train_n, a.val_n, a.batch, a.seed, a.val_from_train)
    ev = dict(epochs=a.epochs, seed=a.seed, seeds=a.seeds, device=a.device)

    log_path = RESULTS / a.out
    log = open(log_path, "w", newline="")
    writer = csv.DictWriter(log, fieldnames=FIELDS)
    writer.writeheader()

    started = time.perf_counter()
    best = seed_genome()
    row = assess(best, tl, vl, **ev)
    writer.writerow({"gen": 0, "cand": 0, "parent": "", "mutation": "seed",
                     "genome": json.dumps(best), **row})
    log.flush()
    best_fit = row["fitness"]
    print(f"gen 0 seed: acc {row['acc']:.4f} params {row['params']:,} "
          f"fitness {best_fit:.4f} ({row['train_s']:.0f}s)")

    if a.random:
        # Random search: every draw is independent of the seed genome and of
        # each other. Logged in the same format so the two runs compare directly.
        for c in range(a.random):
            child = random_genome(rng)
            r = assess(child, tl, vl, **ev)
            writer.writerow({"gen": 1, "cand": c, "parent": "", "mutation": "random",
                             "genome": json.dumps(child), **r})
            log.flush()
            print(f"random {c}: acc {r['acc']:.4f} params {r['params']:,} "
                  f"fitness {r['fitness']:.4f}")
            if r["fitness"] > best_fit:
                best, best_fit = child, r["fitness"]
        a.generations = 0

    for gen in range(1, a.generations + 1):
        improved = False
        for c in range(a.children):
            child, how = mutate(best, rng)
            r = assess(child, tl, vl, **ev)
            writer.writerow({"gen": gen, "cand": c, "parent": "best", "mutation": how,
                             "genome": json.dumps(child), **r})
            log.flush()
            tag = "rejected, undeployable" if not r["deployable"] else (
                f"acc {r['acc']:.4f} params {r['params']:,} fitness {r['fitness']:.4f}")
            print(f"gen {gen} cand {c}: {how:34s} {tag}")
            if r["fitness"] > best_fit:
                best, best_fit, improved = child, r["fitness"], True
        print(f"gen {gen} best fitness {best_fit:.4f}"
              f"{'' if improved else '  (no improvement this generation)'}")

    (RESULTS / log_path.name.replace("search_log", "best_genome").replace(".csv", ".json")
     ).write_text(json.dumps(best, indent=2))
    print(f"\nsearch done in {time.perf_counter() - started:.0f}s, "
          f"best fitness {best_fit:.4f}")
    print(f"-> {log_path}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
