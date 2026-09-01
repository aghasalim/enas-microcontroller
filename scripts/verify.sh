#!/usr/bin/env bash
# Export the network, build the C, and check it against PyTorch end to end.
#
# Run this after changing anything in search/ or export/. A pass means the op
# table, the folded weights and every C kernel still agree with what PyTorch
# computes for the same input.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

python="${PYTHON:-.venv/bin/python}"
if [ ! -x "$python" ]; then
    python="$(command -v python3)"
    echo "note: no .venv, falling back to $python"
fi

echo "== exporting the network from results/best_genome.json =="
"$python" export/export_c.py

echo
echo "== building and running the equivalence test =="
make -C firmware clean >/dev/null
make -C firmware test

echo
echo "== latency on this host =="
cc -std=c99 -O2 -I firmware -c firmware/micronet.c -o /tmp/micronet.o
c++ -std=c++17 -O2 -I firmware firmware/bench.cpp /tmp/micronet.o -o /tmp/micronet_bench
/tmp/micronet_bench firmware/generated 50

echo
echo "all checks passed"
