#!/bin/bash
# PickMoment training.
#
# Usage:
#   bash scripts/train.sh                      # all visible GPUs
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh --wandb
set -e

CONFIG=${CONFIG:-configs/train.yaml}

accelerate launch \
    --mixed_precision bf16 \
    train.py \
    --config "$CONFIG" \
    --seed 42 \
    "$@"
