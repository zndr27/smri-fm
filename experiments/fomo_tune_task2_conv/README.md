# fomo_tune_task2_conv

Two questions, in order, because the second is unanswerable without the first.

**1. What is our NSD?** The challenge ranks task 2 on the mean of the Dice rank and the NSD rank,
and the protocol measured only Dice, so nobody knows. Every task-2 choice on record — the
largest-component filter, the threshold, the checkpoint — was made on half the score.

**2. Does predicting below 8mm help, and which metric does it help?** The baseline emits one
probability per 8mm patch and broadcasts it over all 512 voxels, so its boundary is an 8mm
staircase. Paul Scotti reported on Discord (16 Aug) that replacing that head with a progressive
upsampling conv decoder moves Dice 0.195 -> 0.261 -> 0.313. That work is not in any repo, so this
reproduces it. The interesting part is not the Dice: a staircase surface is roughly half a patch
from a smooth one no matter how large the lesion, so NSD is where the readout should bind hardest.

```bash
./launch.sh task2_linear    # logistic head, at the commit that added NSD to the protocol
./launch.sh task2_conv      # conv decoder, at the commit after
```

Both on `pretrain_full_90_10_h100`, so both compare directly to the `baseline` row of the task 2
leaderboard (Dice 0.195, oracle 0.271).

## Protocol change

`THRESHOLDS` was extended from `np.logspace(-6, -0.3, 60)`, which tops out at 0.501, to reach 1.0.
The old grid suited a patch-fraction readout sitting near the 2.4e-4 voxel prevalence; a decoder
trained with BCE and soft Dice sits near 0.5, and its best cut can be above the old ceiling.

The new grid **contains the old 60 points exactly** (`test_thresholds_contain_the_old_grid`), so
every score already recorded is still reachable and `task2_linear` should reproduce the `baseline`
row rather than supersede it. `AGENTS.md` asks that a protocol edit be a deliberate decision rather
than a step in an experiment — this is the decision, and the superset is what keeps it cheap.

## The baseline does not reproduce on this machine, and that is not the protocol's fault

`task2_linear` scores Dice **0.181**, against **0.195** in the leaderboard's `baseline` row. Before
reading anything else, that gap had to be explained, because a changed protocol is the obvious
suspect.

It is not. Running the unmodified pre-change `main_task2.py` from `11e53ab`, old 60-point grid and
no NSD, on this machine gives `dice=0.1805574242368666`, `dice_oracle=0.2558153938552303`,
`threshold=0.06768750009458542` — every digit identical to `task2_linear`. **The protocol change
moves Dice by exactly zero.**

The gap is hardware and library versions. 21 of 23 folds match the committed H100 log to three
decimals; two flip hard:

| | recorded, H100 | here |
|---|---|---|
| sub-01 | 0.494 | 0.022 |
| sub-10 | 0.400 | 0.530 |

That is the largest-component filter being discontinuous. A hair of difference in the logistic fit
(scikit-learn 1.9 here against the 1.8 pinned in `Apptainer.def`) changes which blob is biggest, and
one subject flipping moves the mean by 0.02 — the whole gap.

**So the conv decoder is compared against 0.181 on this machine, never against the 0.195 row.**

## Results

Point estimate with its 95% bootstrap CI. Deltas are paired per subject against `task2_linear`,
bootstrapped over the same 23 subjects.

| Run | Dice | 95% CI | Oracle | NSD | 95% CI | Oracle |
|---|---|---|---|---|---|---|
| task2_linear | 0.181 | 0.082 – 0.286 | 0.256 | 0.172 | 0.091 – 0.262 | 0.231 |
| task2_conv | 0.258 | 0.139 – 0.392 | 0.319 | 0.196 | 0.099 – 0.301 | 0.286 |
| task2_conv_seed2 | 0.266 | 0.139 – 0.405 | 0.324 | 0.215 | 0.116 – 0.328 | 0.314 |

| Paired vs linear | Dice delta | 95% CI | Dice wins | NSD delta | 95% CI | NSD wins |
|---|---|---|---|---|---|---|
| task2_conv | **+0.078** | +0.010 – +0.159 | 9/23 | +0.024 | -0.053 – +0.100 | 7/23 |
| task2_conv_seed2 | **+0.085** | +0.012 – +0.173 | 10/23 | +0.043 | -0.035 – +0.127 | 9/23 |

Seed spread across the two runs is 0.007 on Dice and 0.019 on NSD, so the Dice gain is an order of
magnitude larger than the seed noise and both its intervals exclude zero. 0.258 and 0.266 also sit
inside the range Paul reported for the same architecture (0.228, 0.261, 0.302, 0.313), which is
about as much agreement as a reconstruction from a chat message can claim.

## The decoder buys Dice. It does not measurably buy NSD.

This is the result worth carrying, and it is the opposite of what the ceiling argument predicts.
Both NSD intervals include zero while both Dice intervals exclude it, on the same runs.

The reason is probably not the readout at all. **NSD is scored at a 1mm tolerance and these
acquisitions have 5.2–6.75mm slices.** Through-plane the boundary is quantized by the scan, not by
the 8mm patch, and no change to the head can move it. A block-level oracle over these masks reaches
NSD 0.449 on a 1mm isotropic grid; scored the way the challenge scores it, the linear head only
manages 0.172, so the patch grid was never the binding constraint on NSD to begin with.

Two subjects make the decoupling concrete — Dice and NSD move in opposite directions:

| | Dice linear -> conv | NSD linear -> conv |
|---|---|---|
| sub-05 | 0.611 -> 0.579 | 0.427 -> **0.119** |
| sub-13 | 0.094 -> 0.121 | 0.243 -> **0.031** |

Better volume overlap, worse boundary placement. Invisible to a Dice-only protocol.

## What is still broken

**12 of 23 subjects score 0.000 Dice under both conv seeds.** The wins are large and concentrated
(sub-01 0.000 -> 0.633, sub-18 0.140 -> 0.687, sub-04 0.533 -> 0.784) rather than spread thin, and
sub-20 regresses 0.253 -> 0.000. Half this task is still a small-lesion detection failure that a
better readout does not touch.

`task2_conv` selects a threshold of 0.857 on fold 13, above the old grid's 0.501 ceiling, so the
threshold extension was load-bearing rather than precautionary.

Both conv runs are single-fold-per-subject with no ensembling. Paul's ensemble and EMA-only numbers
(0.302, 0.313) are above these; neither is reproduced here.
