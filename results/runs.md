# The two runs behind this directory

Two training runs produced everything in `results/`. Neither wrote a
timestamp, so the dates below are the commits that first carried the
files. Every other number is read from a column of the two logs or
from the line of code that set it, and that line is named next to it.

## Run 1, the search

`make search`, which is `controller.py --generations 8 --children 4`
(Makefile, `search` target). Wrote `search_log.csv`, 33 rows, and
`best_genome.json`. Committed 2026-09-01 10:06 +0200 in 5a19196.

| | value | source |
| --- | --- | --- |
| machine | Apple M4, CPU only, macOS 15 | README section 3; requirements.txt:1 |
| threads | 4 | controller.py:75, applied at :83 |
| torch | 2.13.0 | requirements.txt:2 |
| seed | 0, for the weights, the split and the mutation rng | controller.py:74; :51, :40, :85 |
| candidates | 1 seed genome + 8 generations x 4 children = 33 | controller.py:95, :103 to :105 |
| trained | 26 | `deployable` column = 1 |
| rejected before a gradient step | 7 | `deployable` column = 0; controller.py:57 |
| training images | 8,000 from the CIFAR-10 train split | controller.py:71, :38, :41 |
| ranking images | 2,000 from the CIFAR-10 test split | controller.py:72, :39, :42 |
| epochs per candidate | 3 | controller.py:70 |
| batch | 128 train, 256 eval | controller.py:73, :44 |
| optimiser | AdamW, lr 3e-3, weight decay 5e-4 | search/fitness.py:79, :85 |
| schedule | OneCycleLR, max_lr 3e-3, total_steps = 3 x len(loader) | search/fitness.py:86 to :87 |
| gradient clip | 1.0 | search/fitness.py:94 |
| augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip | controller.py:35 |
| normalisation | mean (0.4914, 0.4822, 0.4465), sd (0.2470, 0.2435, 0.2616) | controller.py:33 |
| fitness | acc - 0.02 log10(params) - 0.02 log10(MACs) | search/fitness.py:101, :115 |
| hard caps | 50,000 parameters; 250 KB weights plus peak activation | search/fitness.py:102, :111 to :114 |

### Time

`train_s` is the clock around the three epochs and the eval pass after
them (search/fitness.py:88, :97). It leaves out building the model, the
cost hooks and loading CIFAR-10. A rejected candidate logs 0.0
(controller.py:59).

| | seconds | minutes |
| --- | ---: | ---: |
| sum of `train_s` over 33 rows | 3,878.7 | 64.6 |
| shortest trained candidate, gen 3 cand 3 | 90.7 | |
| longest, gen 8 cand 0 | 227.4 | |
| mean over the 26 trained | 149.2 | |

The same column, one row per generation:

| gen | trained | rejected | `train_s` | best acc in the generation |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0 | 145.6 | 0.4655 |
| 1 | 4 | 0 | 572.3 | 0.4895 |
| 2 | 3 | 1 | 413.6 | 0.4845 |
| 3 | 3 | 1 | 361.4 | 0.4690 |
| 4 | 3 | 1 | 475.7 | 0.5045 |
| 5 | 3 | 1 | 424.0 | 0.5050 |
| 6 | 3 | 1 | 450.0 | 0.4940 |
| 7 | 2 | 2 | 301.5 | 0.4845 |
| 8 | 4 | 0 | 734.6 | 0.5050 |

Generation 8 averages 183.7 s per trained candidate against 149.2 for
the run, at the same parameter count. It is the machine, not the models,
and the log can show that four ways. Generation 8's candidates average
1.97M MACs, fewer than generations 1 to 4 at 2.8M to 5.3M, which trained
faster. Candidate 3 of generation 8 is the same genome as candidate 0 of
generation 5, byte for byte, with the same 0.5050 accuracy, and it took
156.4 s against 148.9 s: identical work, 5% slower. Candidate 2 has the
winner's exact MAC count and took 167.8 s where the winner took 153.4 s.
And the four candidates ran 227, 183, 168 then 156 s in that order,
across four unrelated mutations, which is a machine recovering from
something rather than anything about the architectures. Generation 8
began 52 minutes into 65 of continuous training on a laptop. Whether the
something was thermal or another process, the log cannot say, and it is
not written down anywhere else.

### The same genome, twice

Four genomes appear twice in the `genome` column. `acc` and `fitness`
agree to the last digit each time; `train_s` does not:

| first | again | `acc` | `train_s` |
| --- | --- | ---: | --- |
| gen 0 cand 0, the seed | gen 1 cand 1 | 0.4655 | 145.6 then 149.0 |
| gen 2 cand 1 | gen 2 cand 2 | 0.4845 | 139.3 then 137.1 |
| gen 5 cand 3 | gen 6 cand 2 | 0.4520 | 121.7 then 122.9 |
| gen 5 cand 0 | gen 8 cand 3 | 0.5050 | 148.9 then 156.4 |

That is the determinism evidence README section 3 rests on. The seed
is applied before `build()` (controller.py:51), so initial weights,
subset indices and batch order were all inside seed 0, and an
architecture trained at minute 40 gave the number it gave at minute 5.

## Run 2, the retest

`experiments/validate_winner.py` at its defaults. Wrote
`validation.csv`, 10 rows. Three of them were already in d4e5e21 at
10:46 and the full file landed in 8573a75 at 11:05, 2026-09-01 +0200.

| | value | source |
| --- | --- | --- |
| architectures | the seed genome and `best_genome.json` | experiments/validate_winner.py:70 to :73 |
| training seeds | 0 to 4, both architectures per seed | `seed` column; validate_winner.py:60, :80 to :82 |
| split | validation carved from the train split with seed 1234; test split not opened | validate_winner.py:40, :48 to :53 |
| validation images | 5,000 | validate_winner.py:63 |
| unchanged from the search | 8,000 training images, 3 epochs, batch 128, `train_micro` | validate_winner.py:62, :64, :65, :85 |
| time | `train_s` sums to 1,586.0 s, 26.4 min over 10 trainings | `train_s` column |

## Run 3, the 5 seed rerun on a GPU

`make search-gpu` on a rented RunPod RTX 3090, community cloud, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, torch
2.4.1+cu124. Run on 2026-09-26 between 23:27 and 23:48 UTC. Two
commands (Makefile, `search-gpu` target):

| | hill climb | random search |
| --- | --- | --- |
| command | `controller.py --generations 8 --children 4 --seeds 5 --device cuda --val-from-train` | `controller.py --random 33 --seeds 5 --device cuda --val-from-train` |
| log | `search_log_gpu.csv` | `random_log_gpu.csv` |
| rows | 33 | 34, the seed genome and 33 draws |
| winner | `best_genome_gpu.json` | `best_genome_random_gpu.json`, taken from the log |
| time | `train_s` sums to 573 s | `train_s` sums to 664 s |

Seeds 0 to 4 for every candidate, ranked on the mean, the per seed
accuracies in the `accs` column. Validation is the first 2,000 of a
seed 0 permutation of the train split and training the next 8,000
(controller.py, `loaders`). The test split is not opened. Console
output in `run_gpu.log`.

## Run 4, the fresh seed rescore

`experiments/validate_winner.py --no-winner --first-seed 100 --seeds 5
--arch hill_gpu=results/best_genome_gpu.json --arch
random_gpu=results/best_genome_random_gpu.json --out rescore_gpu.csv`,
on the Apple M4 CPU with 4 threads, torch 2.13.0. Seeds 100 to 104,
which neither search used, on the split seed 1234 carve from Run 2.
15 trainings, `rescore_gpu.csv`.

## Run 5, two more search seeds on a GPU

Same command as Run 3's hill climb with `--seed 1` and `--seed 2`, on
an RTX A5000 (secure cloud) on 2026-09-27. The seed sets the mutation
rng, the training seeds (seed to seed + 4) and the validation carve, so
each is an independent search. Logs `search_log_gpu_seed1.csv` and
`search_log_gpu_seed2.csv`, winners `best_genome_gpu_seed1.json` and
`best_genome_gpu_seed2.json`, console output in `run_gpu_seed1.log` and
`run_gpu_seed2.log`.

## Run 6, full training

`export/train_winner.py --genome <genome> --device cuda --seed <s>`, on
the same A5000, for the baseline, the Run 1 CPU winner and the Run 3
GPU winner, seeds 0 to 2. 12 epochs on 40,000 training images, scored on
5,000 held out of the training split with the split seed 1234 carve
(train_winner.py defaults). Every row is in `full_train.csv`. The GPU
winner at seed 0 is `winner.pt`, which is what section 7 exports.

## Run 7, rescoring the new winners

As Run 4, on the Mac CPU, for the Run 5 winners:
`validate_winner.py --no-winner --first-seed 100 --seeds 5 --arch
hill_gpu_seed1=... --arch hill_gpu_seed2=... --out
rescore_gpu_seeds12.csv`.

## Weights

`winner.pt` is the only checkpoint committed, 200 KB, the network the
firmware runs. The other eight full training checkpoints came back from the
pod but are not committed.
