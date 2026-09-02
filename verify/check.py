# Recompute search-log aggregates in Python and check them against README.md.
#
# Covers the same subset as the SQL verifier: candidate counts, training
# minutes, best fitness/accuracy, and the seed-vs-best comparison.

import csv, math, sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "."

with open(os.path.join(root, "results", "search_log.csv"), newline="") as f:
    rows = list(csv.DictReader(f))

readme = open(os.path.join(root, "README.md"), encoding="utf-8").read()

# Cast fields once.
for r in rows:
    r["params"]     = int(r["params"])
    r["macs"]       = int(r["macs"])
    r["peak_act"]   = int(r["peak_act"])
    r["acc"]        = float(r["acc"])
    r["fitness"]    = float(r["fitness"]) if r["fitness"] != "-inf" else float("-inf")
    r["deployable"] = int(r["deployable"])
    r["train_s"]    = float(r["train_s"])

ok   = [r for r in rows if r["deployable"] == 1]
seed = rows[0]
best = max(ok, key=lambda r: r["fitness"])

# Duplicated genomes: a genome seen more than once.
from collections import Counter
genome_counts = Counter(r["genome"] for r in rows)
dup_count = sum(1 for c in genome_counts.values() if c > 1)

# Training minutes.
total_train_min = round(sum(r["train_s"] for r in rows) / 60.0)

# Duplicate training minutes: rows whose genome appeared in an earlier row.
seen = {}
dup_train_s = 0.0
for i, r in enumerate(rows):
    if r["genome"] in seen:
        dup_train_s += r["train_s"]
    else:
        seen[r["genome"]] = i
dup_train_min = round(dup_train_s / 60.0)

# Insert operator stats.
insert_rows = [r for r in rows if r["mutation"].startswith("insert")]
insert_trainable = sum(r["deployable"] for r in insert_rows)

# Accuracy gap.
gap_points = 100 * (best["acc"] - seed["acc"])

# Parameter difference.
param_diff = best["params"] - seed["params"]

# MAC reduction percentage.
mac_pct = 100 * (1.0 - best["macs"] / seed["macs"])

# Working set in KB.
best_kb = (best["params"] + best["peak_act"]) / 1024.0
worst_kb = max((r["params"] + r["peak_act"]) / 1024.0 for r in ok)
headroom = 250.0 / worst_kb

# Candidate more accurate than the best-by-fitness.
more_acc = [r for r in ok if r["acc"] > best["acc"]]
acc_rank = len(more_acc) + 1
if more_acc:
    top_acc = max(more_acc, key=lambda r: r["acc"])

# Which generations improved: a generation improved if one of its deployable
# children beat every fitness seen before it in the log.
best_so_far = float("-inf")
improved = []
for r in ok:
    if r["gen"] != "0" and r["fitness"] > best_so_far:
        if r["gen"] not in improved:
            improved.append(r["gen"])
    if r["fitness"] > best_so_far:
        best_so_far = r["fitness"]

want = [
    f"{len(rows)} candidates",
    f"{len(rows) - 1} child slots",
    f"{sum(1 for r in rows if r['deployable'] == 0)} rejected",
    f"{dup_count} evaluated twice",
    f"{total_train_min} minutes of training",
    f"{dup_train_min} minutes of duplicated",
    f"{len(ok)} trained candidates",
    f"drawn {len(insert_rows)} times and produced {insert_trainable} trainable",
    f"**{best['acc']:.4f}**",
    f"**{best['fitness']:.4f}**",
    f"{gap_points:.1f} accuracy points",
    f"{param_diff} more",
    f"{mac_pct:.1f}% fewer",
    f"generation {best['gen']}, candidate {best['cand']}",
    f"{best_kb:.1f} KB",
    f"{worst_kb:.1f} KB",
    f"{headroom:.1f}x",
    f"{acc_rank}th by accuracy",
]

failures = 0
for x in want:
    hit = x in readme
    tag = "ok" if hit else "FAIL"
    print(f"  {tag:<4} {x}")
    if not hit:
        failures += 1

if failures > 0:
    print(f"{failures} of {len(want)} figures are not in README.md as written")
    sys.exit(1)
print(f"Python reproduces all {len(want)} figures from the search log")
