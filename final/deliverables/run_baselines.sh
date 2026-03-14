#!/usr/bin/env bash
# DATA 37100 — Final Project: Baseline Runs
# Run from the REPO ROOT: bash final/draft/run_baselines.sh
#
# Produces two baseline runs:
#   1. Diffusion (MNIST, T=200, eps target)
#   2. DCGAN (MNIST, 400 steps)
#
# Outputs go to ./untrack/outputs/final/{diffusion,gan}/

set -euo pipefail

echo "============================================"
echo "  Baseline 1: Diffusion (MNIST, T=200, eps)"
echo "============================================"
python final/starter/src/diffusion_baseline.py \
    --dataset mnist \
    --download \
    --epochs 1 \
    --T 200 \
    --target eps \
    --base-ch 64 \
    --seed 42 \
    --device auto

echo ""
echo "============================================"
echo "  Baseline 2: DCGAN (MNIST, 400 steps)"
echo "============================================"
python final/starter/src/gan_baseline.py \
    --dataset mnist \
    --download \
    --epochs 1 \
    --max-steps 400 \
    --batch-size 128 \
    --lr 0.0002 \
    --d-steps 1 \
    --base-ch 64 \
    --z-dim 128 \
    --device auto

echo ""
echo "Done. Check ./untrack/outputs/final/ for outputs."
