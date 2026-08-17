#!/usr/bin/env bash
# Run locally on one GPU, not on the cluster: 23 subjects, well inside half an hour.
#
#   ./launch.sh task2_linear    # at the commit where Task2Method is still the logistic head
#   ./launch.sh task2_conv      # at the commit where it is the conv decoder
#
# The two runs are two code states, not two configs, so the run name is an argument.
set -euo pipefail

name="$1"

cd "$(dirname "$0")/../.."

export HF_HOME="${HF_HOME:-/mnt/data/medarc/neuro-fm/.hf_cache}"

OUT_DIR="experiments/fomo_tune_task2_conv/output"

# the default checkpoint, so both rows compare to the `baseline` row of the task 2 leaderboard
CKPT="hf://medarc/walnut/checkpoints/pretrain_full_90_10_h100/checkpoint-last.pth"

uv run python -m fomo_tune.main_task2 train \
    ckpt_path="${CKPT}" \
    output_root="${OUT_DIR}" \
    name="${name}"

cat "${OUT_DIR}/${name}/metrics.json"
