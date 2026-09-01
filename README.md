# ENAS for microcontrollers

**An evolutionary architecture search under a hard 50,000 parameter cap, that
logs the candidates it rejected as well as the one it kept.**

[![ci](https://github.com/aghasalim/enas-microcontroller/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/enas-microcontroller/actions/workflows/ci.yml)
[![licence](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

A hand-written baseline for a Cortex-M class device is mutated one edit at a
time, each child is trained briefly on CIFAR-10 and scored, and anything that
would not fit in the device budget is rejected before a single gradient step is
spent on it. Every candidate the search ever saw, including the 7 it threw away
and the 4 it accidentally evaluated twice, is a row in
[`results/search_log.csv`](results/search_log.csv). Every number below and every
figure is derived from that one file, and [`check_numbers.py`](check_numbers.py)
fails the build if the two disagree.

## The result

| | accuracy | parameters | MACs | peak activation | fitness |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed, hand written | 0.4655 | 48,354 | 1,920,768 | 12,288 | 0.2461 |
| **best found** | **0.5025** | 48,546 | **1,761,024** | 12,288 | **0.2839** |

3.7 accuracy points and 8.3% fewer multiply-accumulates, for 192 more
parameters. Found at generation 5, candidate 2, after 65 minutes of training
across 33 candidates.

**Read 0.5025 as a ranking score, not as an accuracy.** Every candidate got 3
epochs on an 8,000 image subset, which is 16% of the CIFAR-10 training set, at
under 50,000 parameters. That budget is what makes a 33 candidate search cost an
hour instead of a week. It is enough signal to order architectures and nowhere
near enough to state what the winner is worth. The winner has not been retrained
on the full training set yet, so this repository does not contain a headline
accuracy and does not claim one.

![search trajectory](results/figures/search-trajectory.png)

## What the search actually changed

![seed against winner, block by block](results/figures/topology.png)

Two edits separate the winner from the baseline: block 0 changed from a
zero-parameter shift block to a learned asymmetric depthwise block, and block 4
gained a stride. That is the whole difference.

The path there was not two steps. Block 3's stride went 2 to 1 in generation 1,
which was the first improvement the search found, and then back to 2 in
generation 5, which was the last. Four generations were spent on a change that
was eventually undone. The trajectory figure shows this as the long flat stretch
between generation 1 and generation 4.

## Three things the log supports

**Learned spatial filtering pays at high resolution and not at low.** In
generation 4 both conversions were drawn from the same parent, in the same
generation, so this is a controlled pair rather than two anecdotes:

| edit | position | resolution | accuracy | change |
| --- | --- | --- | ---: | ---: |
| `kind block0 shift->asym` | first block | 16x16 | 0.5045 | +0.0150 |
| `kind block4 shift->asym` | last block | 4x4 | 0.4825 | -0.0070 |

Parent accuracy was 0.4895. The last-block conversion was drawn again in
generation 8 from a stronger parent and lost again (0.5005 against 0.5025), so
the direction held twice.

**The gain came from having learned filters early, not from a bigger receptive
field.** Once block 0 was a learned block, widening its kernel from 3 to 7 in
generation 7 dropped accuracy from 0.5025 to 0.4695. A larger footprint on the
same block made it worse, so the 3x3 is doing the work.

**The parameter cap bound the search. The SRAM budget never did.**

![budget](results/figures/budget.png)

All 7 rejections were the 50,000 parameter cap. Not one candidate came close to
the 250 KB SRAM budget: the worst working set observed, weights plus peak
activation at int8, was 69.5 KB, and the winner sits at 59.4 KB. The search
spent the whole run pressed against one constraint while the other had 3.6x
headroom it never used. The obvious follow-up is to raise the parameter cap to
where the two bind together and rerun.

## What the search wasted

![operator yield](results/figures/operators.png)

**`insert` was drawn 5 times and produced 0 trainable children.** At roughly
48,500 parameters against a 50,000 cap, adding a block cannot fit, so the
operator was dead weight for the entire run. It cost no training time, because
the budget check runs before training, but it burned 5 of the 32 child slots. A
fix is to pair an insertion with a compensating width reduction so the child is
born inside the budget.

**4 genomes were evaluated twice, for 9 minutes of duplicated training.** The
mutation operator retries when a child equals its parent, but it never compares
against everything already seen, so the search re-derived four known points. A
memo keyed on the genome would remove this.

Those duplicates did pay for one thing. All four pairs returned byte-identical
accuracy, including two different mutation labels that happened to arrive at the
same architecture, which is four independent confirmations that the run is
deterministic. That comes from seeding before the model is constructed rather
than after, which is the ordering that makes weight initialisation fall inside
the seed.

**5 of the 8 generations produced no improvement.** All the gain is in
generations 1, 4 and 5.

## The fitness function

```
F = Acc - 0.02 * log10(params) - 0.02 * log10(MACs)
```

subject to two hard rejections rather than penalties: over 50,000 parameters, or
weights plus peak activation over 250 KB at int8. A model that breaches either
one cannot be flashed at all, so it scores negative infinity instead of a good
accuracy with a deduction. A fitness function that lets an undeployable model
win is measuring the wrong thing.

Peak activation is measured with forward hooks, as the largest single tensor any
module produces, not estimated from the parameter count. Weights are the number
people quote; activations are the number that makes the deployment fail. A real
arena allocator does better than this by reusing buffers, so treat 69.5 KB as an
upper bound on the working set rather than a prediction.

The log ranks the winner 4th by accuracy. Two candidates reached 0.5050 and one
reached 0.5045, all with more MACs, and the MAC term preferred 0.5025 at
1,761,024 MACs over 0.5050 at 2,570,496. That trade is the fitness function
doing what it was written to do, and it is visible in the log rather than hidden
behind a single reported winner.

## The search space

A genome is a dict, not generated code:

```python
{"stem_width": 24, "act_shift": 4,
 "blocks": [{"kind": "shift", "out": 48, "stride": 1, "k": 3}, ...]}
```

so a mutation is an edit that can be logged, diffed and replayed, and every
architecture in the run is reconstructible from its genome alone. Replaying the
33 logged mutations in order reproduces
[`results/best_genome.json`](results/best_genome.json) exactly, which is checked
by a test.

Seven operators: `width`, `kernel`, `kind`, `act`, `insert`, `drop`, `stride`.
One edit per child, so a fitness change is attributable to one cause. `kernel`
only targets asymmetric blocks, because shift blocks have no spatial weights and
offering them a kernel edit produces a duplicate model.

Two block types, both chosen for what they cost on the device:

- **`shift`**, from [`search/blocks.py`](search/blocks.py), replaces the spatial
  half of a convolution with `torch.roll` on five channel groups. No weights, no
  multiplies, and on the device it is pointer arithmetic.
- **`asym`** buys a k x k depthwise receptive field as 1 x k then k x 1. Ten
  weights per channel instead of 25 at k=5, and the intermediate never
  materialises at full width, which matters more for peak SRAM than the weight
  saving does.

The activation is a ReLU6 with a leak of 2^-shift, which is an arithmetic right
shift in an int8 pipeline rather than a multiply, and the clamp at 6 keeps the
output range fixed so the activation folds into requantisation.

One caveat on the activation. The `act` operator was drawn 4 times. Against the
generation 1 parent the default 2^-4 beat both 2^-2 (0.4640) and 2^-5 (0.4815),
and against the final parent it beat 2^-5 again (0.4905 against 0.5025). But
2^-3 was only ever tried against the weakest parent in the run, the seed, where
it won (0.4790 against 0.4655). So 2^-3 is untested on any strong topology, and
the default surviving is weaker evidence than a clean sweep would be.

## Reproducing

```bash
make setup
make search     # 65 min on an M4 CPU, downloads CIFAR-10 on first run
make figures
make check
```

`make search` writes `results/search_log.csv` and `results/best_genome.json`.
`make figures` redraws every figure above from that log and trains nothing.
`make check` re-derives the numbers in this README from the log and exits
non-zero on the first disagreement.

The run is CPU only and fixed to 4 threads. These models are small enough that
thread overhead costs more than the extra parallelism returns, which was
measured before the run rather than assumed. Per-candidate wall clock is in the
`train_s` column.

## What this is not

It is not a state of the art result and it is not a comparison against MCUNet or
any published NAS method, because the training budget here is far too short for
such a comparison to mean anything. It is a search that ran to completion under
a real deployment constraint, with its failures in the log.

The 33 candidate budget is small. With 8 generations, 4 children each, and one
edit per child, the search covers a thin path through the space rather than
mapping it, and a different seed would very likely land somewhere else. Nothing
here separates what the search found from what one run of a hill climber with 26
trained samples happens to find.

Nothing in this repository was tested on hardware. Parameters, MACs and peak
activation are measured from the PyTorch graph. The int8 footprint assumes one
byte per weight and per activation, which is what CMSIS-NN gives you, but no
model here has been quantised, exported or flashed, so every KB figure is a
projection from the float graph and not a measurement from a device.
