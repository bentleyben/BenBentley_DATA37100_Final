# DATA 37100 Final Project — Draft

**Student:** Ben Bentley
**Research Question:** How do diffusion timesteps (T) and prediction target (eps vs x0) affect sample sharpness and failure modes on MNIST?

---

## Quick Start

```bash
# Navigate to repo root
cd "/Users/benbentley/Documents/School/UChicago/Winter 2026/Intro to AI Deep Learning and Generative AI/BenBentley_DATA37100_Final"

# Run baselines (Diffusion + DCGAN)
bash final/draft/run_baselines.sh

# Run controlled experiment (T × target grid)
bash final/draft/run_experiment.sh

# Analyze results
jupyter notebook final/draft/analysis.ipynb
```

**Expected total runtime:** ~15-30 minutes (depending on hardware)

---

## Project Structure

```
final/draft/
├── README.md              # This file
├── run_baselines.sh       # Runs 2 baseline models (Diffusion + DCGAN)
├── run_experiment.sh      # Runs 6-run grid experiment (T × target)
├── analysis.ipynb         # Jupyter notebook for visualization and analysis
└── report.md              # Final report (~3-5 pages)

final/starter/src/         # Provided starter code
├── diffusion_baseline.py  # Diffusion model training script
├── gan_baseline.py        # DCGAN training script
├── transformer_baseline.py
└── utils_data.py          # Data loading utilities

final/tools/
└── visualize_samples.py   # Contact sheet generator

untrack/outputs/final/     # Output directory (not committed)
├── diffusion/             # Diffusion runs
│   ├── results.csv        # Experiment manifest
│   └── run_YYYYMMDD_HHMMSS_*/  # Individual runs
│       ├── run_args.json
│       ├── summary.json
│       └── *.png          # Sample grids
└── gan/                   # GAN runs
    └── run_YYYYMMDD_HHMMSS_*/
```

---

## Setup

### 1. Environment

**Required:**
- Python 3.8+
- PyTorch 1.13+
- torchvision
- numpy
- matplotlib
- pandas
- jupyter (for analysis)

**Install dependencies:**
```bash
# Option 1: pip
pip install torch torchvision numpy matplotlib pandas jupyter

# Option 2: conda
conda install pytorch torchvision numpy matplotlib pandas jupyter -c pytorch
```

### 2. Data

The scripts will automatically download MNIST to `./data/bigdata/MNIST/` on first run (no manual setup needed).

---

## Usage

### Run Baselines (Required)

Produces two working baselines for model families:

```bash
bash final/draft/run_baselines.sh
```

**Outputs:**
- Diffusion baseline: `./untrack/outputs/final/diffusion/run_*/`
- DCGAN baseline: `./untrack/outputs/final/gan/run_*/`

**What's trained:**
1. **Diffusion:** MNIST, T=200, eps target, 1 epoch (~5-10 min)
2. **DCGAN:** MNIST, 400 steps (~3-5 min)

### Run Controlled Experiment (Required)

Runs a 6-run grid experiment (T × target):

```bash
bash final/draft/run_experiment.sh
```

**Experiment design:**
- **Knob 1:** T ∈ {100, 200, 400}
- **Knob 2:** target ∈ {eps, x0}
- **Total runs:** 6
- **Runtime:** ~10-20 minutes total

**Outputs:**
- Run directories: `./untrack/outputs/final/diffusion/run_*/`
- Results manifest: `./untrack/outputs/final/diffusion/results.csv`
- Contact sheet: `./untrack/outputs/final/diffusion/contact_sheet.png`

### Analyze Results

Open the Jupyter notebook:

```bash
jupyter notebook final/draft/analysis.ipynb
```

The notebook will:
1. Load experiment results from `results.csv`
2. Display sample grids side-by-side
3. Compare baseline performance (Diffusion vs DCGAN)
4. Analyze T × target interaction effects
5. Identify failure modes

---

## Hardware Requirements

**Minimum:**
- CPU: Any modern CPU (will be slow)
- RAM: 8GB
- Disk: 500MB for outputs
- Runtime: ~30-45 minutes

**Recommended:**
- GPU: CUDA-capable GPU (e.g., NVIDIA GTX 1060+)
- RAM: 16GB
- Disk: 1GB
- Runtime: ~10-15 minutes

**Notes:**
- The scripts use `--device auto` which selects GPU if available, otherwise CPU
- Training is intentionally kept small (1 epoch, MNIST) to enable fast iteration
- If you encounter OOM errors, reduce `--batch-size` in the scripts

---

## Deliverables

### A. Technical Analysis (Required)

1. **Code:** All scripts in `final/draft/` are runnable
2. **Outputs:** Sample grids saved in `./untrack/outputs/final/`
3. **Analysis:** `analysis.ipynb` contains visualization and interpretation

### B. Repository Hygiene (Required)

1. **README:** This file (setup, run commands, hardware notes)
2. **No large data:** MNIST is downloaded on-the-fly (not committed)
3. **Outputs in untrack/:** All outputs go to `./untrack/` (gitignored)

### C. Summary Report (Required)

**File:** `report.md` (~3-5 pages)

**Contents:**
1. Research question + motivation
2. Methods (baselines + experiment design)
3. Results (figures + sample grids)
4. Failure modes + limitations
5. Conclusions

---

## Troubleshooting

### Issue: "No module named 'lab07_diffusion_core'"

**Solution:** Run scripts from the **repo root**, not from `final/draft/`:
```bash
cd "/Users/benbentley/Documents/School/UChicago/Winter 2026/Intro to AI Deep Learning and Generative AI/DATA37100-26Win"
bash final/draft/run_baselines.sh
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size in the scripts:
```bash
# Edit run_baselines.sh or run_experiment.sh
# Change --batch-size 128 to --batch-size 64 or --batch-size 32
```

### Issue: "results.csv not found"

**Solution:** Run the experiment first:
```bash
bash final/draft/run_experiment.sh
```

### Issue: Slow training on CPU

**Expected behavior:** Training on CPU will be 5-10× slower than GPU. The full experiment may take 30-60 minutes.

**Solution:** If you have a CUDA GPU, install PyTorch with CUDA:
```bash
# Check CUDA version first
nvidia-smi

# Install PyTorch with CUDA (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Reproducibility

**Controlled variables:**
- Seed: 42 (fixed across all runs)
- Dataset: MNIST
- Architecture: UNet (base_ch=64) for diffusion, standard DCGAN for GAN
- Training: 1 epoch

**Variable factors (experiment):**
- T ∈ {100, 200, 400}
- target ∈ {eps, x0}

**To reproduce exact results:**
1. Use the same hardware (CPU vs GPU may produce slight numerical differences)
2. Use the same PyTorch version (tested with PyTorch 2.0+)
3. Run from repo root with provided scripts

---

## Model Coverage (Meets Requirements)

- ✅ **Diffusion** (Week 7): Baseline + controlled experiment
- ✅ **DCGAN** (Week 4): Baseline
- Total: **2 model families** (satisfies "at least two" requirement)

---

## Contact

**Student:** Ben Bentley
**Course:** DATA 37100 (Winter 2026)
**Instructor:** [Course instructor]

For questions about this project, see the final report (`report.md`) or analysis notebook (`analysis.ipynb`).
