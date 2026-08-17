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

## Results

Pending.
