PY := .venv/bin/python
PIP := .venv/bin/pip
PYTHON312 ?= python3.12

.PHONY: setup test search figures check clean

setup:
	$(PYTHON312) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PY) -m pytest -q

# The full run in the README: 8 generations, 4 children each, 65 minutes on an
# M4 with 4 threads. Downloads CIFAR-10 into data/ on the first call.
search:
	$(PY) controller.py --generations 8 --children 4

# The GPU rerun that answers section 6: five seeds per candidate, validation
# held out of the training split, and a random search baseline of the same
# size as the hill climb (1 + 8 x 4 = 33 candidates). Writes separate files, the
# CPU run above stays untouched.
search-gpu:
	$(PY) controller.py --generations 8 --children 4 --seeds 5 --device cuda --val-from-train --out search_log_gpu.csv
	$(PY) controller.py --random 33 --seeds 5 --device cuda --val-from-train --out random_log_gpu.csv

# Redraws every README figure from results/search_log.csv. Trains nothing.
figures:
	$(PY) experiments/make_figures.py

# Fails if any number quoted in the README disagrees with the search log.
check:
	$(PY) check_numbers.py

clean:
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
