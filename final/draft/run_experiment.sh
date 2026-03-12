#!/usr/bin/env bash
# DATA 37100 — Final Project: Controlled Experiment (Two-Knob Grid)
# Run from the REPO ROOT: bash final/draft/run_experiment.sh
#
# Two-knob study:
#   Knob 1: T (diffusion timesteps) ∈ {100, 200, 400}
#   Knob 2: target (prediction parameterization) ∈ {eps, x0}
#   Total runs: 3 × 2 = 6
#
# Question: How do diffusion timesteps and prediction target
#           affect sample sharpness and failure modes on MNIST?
#
# Outputs: ./untrack/outputs/final/diffusion/

set -euo pipefail

echo "============================================"
echo "  Two-Knob Experiment: T × target"
echo "  T ∈ {100, 200, 400}"
echo "  target ∈ {eps, x0}"
echo "  Total: 6 runs"
echo "============================================"

python final/starter/src/diffusion_baseline.py \
    --dataset mnist \
    --download \
    --epochs 1 \
    --base-ch 64 \
    --seed 42 \
    --device auto \
    --grid "T=100,200,400;target=eps,x0"

echo ""
echo "============================================"
echo "  Grid experiment complete."
echo "  Results manifest: ./untrack/outputs/final/diffusion/results.csv"
echo "============================================"

# Visualize all runs in a contact sheet
echo "Generating contact sheet..."
python final/tools/visualize_samples.py \
    --results-csv ./untrack/outputs/final/diffusion/results.csv \
    --ncols 3

echo "Done."
