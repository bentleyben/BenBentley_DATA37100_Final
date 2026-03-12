# DATA 37100 Final Project Report

**Student:** Ben Bentley
**Date:** March 2026
**Course:** DATA 37100 — Intro to AI, Deep Learning, and Generative AI (Winter 2026)

---

## 1. Research Question & Motivation

### Question
**How do diffusion timesteps (T) and prediction target (eps vs x0) affect sample sharpness and failure modes on MNIST?**

### Motivation
Diffusion models have two critical hyperparameters that govern the noise-to-data trajectory:
- **T (timesteps):** Controls the granularity of the denoising process. More timesteps allow finer-grained noise removal but increase computational cost.
- **target (prediction parameterization):** Determines whether the model predicts the noise (`eps`) or the clean image (`x0`). This choice affects training stability and sample quality.

Understanding how these factors interact is essential for:
1. Efficiently tuning diffusion models for new tasks
2. Diagnosing failure modes (blur, artifacts, instability)
3. Choosing the right trade-off between speed and quality

This study provides a controlled investigation of both factors on a simple, interpretable dataset (MNIST), enabling clean causal inference.

---

## 2. Methods

### 2.1 Model Families (Baselines)

**Baseline 1: Diffusion Model**
- Dataset: MNIST (28×28 grayscale digits)
- Architecture: UNet with sinusoidal time embeddings
- Base channels: 64
- Training: 1 epoch (~60k samples)
- Noise schedule: Linear β schedule
- Default config: T=200, target=eps

**Baseline 2: DCGAN**
- Dataset: MNIST
- Architecture: Standard DCGAN (Generator + Discriminator)
- Base channels: 64
- Latent dimension: 128
- Training: 400 steps
- Learning rate: 0.0002 (Adam)
- Discriminator steps per generator step: 1

*Rationale for choosing these models:* Both are image generative models, enabling direct visual comparison. Diffusion provides a stable, predictable baseline; DCGAN demonstrates the GAN failure mode (mode collapse).

### 2.2 Controlled Experiment (Two-Knob Grid)

**Experimental Design:**
- **Knob 1:** T ∈ {100, 200, 400}
- **Knob 2:** target ∈ {eps, x0}
- **Total runs:** 3 × 2 = 6

**Control variables:**
- Dataset: MNIST
- Architecture: UNet (base_ch=64, time_emb_dim=256)
- Training: 1 epoch
- Seed: 42
- Noise schedule: Linear β ∈ [1e-4, 0.02]

**Hardware:**
- *[Fill in: CPU/GPU type]*
- Runtime per run: *[Fill in after running experiments]*

**Evaluation:**
- Visual inspection of 8×8 sample grids
- Qualitative assessment of sharpness, diversity, artifacts
- Identification of failure modes

---

## 3. Results

### 3.1 Baseline Performance

#### Diffusion (T=200, eps)
*[Insert 8×8 sample grid image]*

**Observations:**
- *[Describe sample quality: sharpness, diversity, recognizability]*
- Runtime: *[X]* seconds
- *[Note any obvious artifacts or failures]*

#### DCGAN (400 steps)
*[Insert 8×8 sample grid image]*

**Observations:**
- *[Describe sample quality compared to diffusion]*
- Runtime: *[X]* seconds
- *[Check for mode collapse: are all 10 digits present?]*

**Baseline Comparison:**
- *[Which baseline produces sharper samples?]*
- *[Which is faster?]*
- *[Which shows more diversity?]*

---

### 3.2 Effect of T (Timesteps)

*[Insert side-by-side grids for T=100, 200, 400 (fixed target=eps)]*

| T   | Visual Quality                | Runtime |
|-----|-------------------------------|---------|
| 100 | *[Describe: blurry? sharp?]*  | *[X]s*  |
| 200 | *[Describe]*                  | *[X]s*  |
| 400 | *[Describe]*                  | *[X]s*  |

**Key Findings:**
1. *[Does increasing T improve sharpness?]*
2. *[Is there a point of diminishing returns?]*
3. *[Any artifacts introduced at high T?]*

**Mechanistic Interpretation:**
- *[Why does T affect quality? Relate to the discrete approximation of the continuous SDE]*

---

### 3.3 Effect of target (eps vs x0)

*[Insert side-by-side grids for target=eps vs x0 (fixed T=200)]*

| Target | Visual Quality                        | Stability |
|--------|---------------------------------------|-----------|
| eps    | *[Describe: sharpness, artifacts]*    | *[Stable/unstable]* |
| x0     | *[Describe]*                          | *[Stable/unstable]* |

**Key Findings:**
1. *[Which target produces sharper samples?]*
2. *[Which shows more artifacts (if any)?]*
3. *[Any difference in digit recognizability?]*

**Mechanistic Interpretation:**
- `eps` predicts noise → indirectly estimates x0 via reparameterization
- `x0` predicts clean image directly → may be more sensitive to training instability
- *[Which interpretation fits your observations?]*

---

### 3.4 Interaction Effects (T × target)

*[Insert full 3×2 grid: rows=T, columns=target]*

**Question:** Does the effect of T depend on the choice of target?

**Observations:**
- At T=100: *[Is eps or x0 better? Why?]*
- At T=400: *[Does the gap between eps and x0 widen or narrow?]*
- *[Any non-linear interaction? E.g., x0 improves more with T than eps does?]*

**Conclusion:**
- *[Independent effects or interaction?]*

---

## 4. Failure Modes (Required)

### 4.1 Mode Collapse in DCGAN

**Evidence:**
*[Check DCGAN samples: count unique digit classes visible in the 8×8 grid]*

**Observed failure:**
- *[E.g., "Only 3 digit classes (1, 3, 7) appear repeatedly; classes 0, 2, 4, 5, 6, 8, 9 are missing"]*

**Likely cause:**
- Adversarial training instability — generator exploits discriminator weaknesses by specializing on a few "easy" modes
- Discriminator fails to push back on low-diversity generations

**Implications:**
- GANs require careful tuning (learning rates, discriminator steps, regularization) to avoid collapse
- Diffusion models do not suffer from this failure mode (likelihood-based training)

---

### 4.2 Blurry Samples at Low T (Diffusion)

**Evidence:**
*[Compare T=100 vs T=400 samples side-by-side]*

**Observed failure:**
- *[E.g., "T=100 samples are noticeably blurrier; digit edges are soft and strokes are thick"]*

**Likely cause:**
- Discrete approximation error — the SDE assumes infinitesimal steps, but T=100 forces large jumps
- Insufficient denoising capacity — each step removes "too much" noise, overshooting the true posterior

**Implications:**
- Speed-quality trade-off: T=100 is ~4× faster than T=400 but produces worse samples
- For deployment, need to balance inference cost vs perceptual quality

---

### 4.3 Other Artifacts (if observed)

**Observed:**
- *[E.g., checkerboard patterns, color shifts, structural collapse, digit deformation]*

**Hypothesis:**
- *[Upsampling artifacts from transpose convolutions?]*
- *[Numerical instability in noise schedule at extreme T?]*
- *[Overfitting due to only 1 epoch?]*

---

## 5. Limitations

1. **Single epoch training:** Models may not be fully converged; longer training could change conclusions
2. **Single dataset (MNIST):** Results may not generalize to natural images (CIFAR-10, ImageNet) or text
3. **Fixed architecture:** Did not vary model width, depth, or attention mechanisms
4. **No quantitative metrics:** Relied on visual inspection; FID or IS scores could provide objective comparisons
5. **Limited T range:** Did not test very low (T=10) or very high (T=1000) extremes

---

## 6. Conclusions

### Key Findings
1. *[Main result about T's effect — e.g., "Increasing T from 100 to 400 significantly improves sharpness but shows diminishing returns beyond 200"]*
2. *[Main result about target — e.g., "eps and x0 produce comparable quality at T=200, but x0 degrades more at low T"]*
3. *[Comparison — e.g., "Diffusion produces more diverse samples than DCGAN, which suffers from mode collapse"]*

### One Surprising Result
- *[Something unexpected — e.g., "x0 target at T=400 produced sharper samples than eps, contrary to expectation that direct prediction would be less stable"]*

### One Next Step
- *[Proposed follow-up — e.g., "Run a second experiment varying noise schedule (linear vs cosine) to see if schedule interacts with T"]*
- *[Or: "Train for 5 epochs to check if conclusions hold at convergence"]*

---

## 7. Reproducibility

**Run commands:**
```bash
# From repo root
bash final/draft/run_baselines.sh    # Produces diffusion + GAN baselines
bash final/draft/run_experiment.sh   # Produces 6-run grid experiment
```

**Expected runtime:**
- Baselines: ~*[X]* minutes total
- Experiment grid: ~*[X]* minutes total
- Analysis: ~*[X]* hours

**Hardware assumptions:**
- *[CPU/GPU type]*
- 16GB RAM minimum
- ~500MB disk for outputs

**Outputs:**
- Sample grids: `./untrack/outputs/final/{diffusion,gan}/*/samples/*.png`
- Run logs: `./untrack/outputs/final/{diffusion,gan}/*/summary.json`
- Results manifest: `./untrack/outputs/final/diffusion/results.csv`

---

## Appendix: Code Structure

```
final/draft/
├── run_baselines.sh      # Runs diffusion + GAN baselines
├── run_experiment.sh     # Runs 6-run grid (T × target)
├── analysis.ipynb        # Jupyter notebook with visualizations
├── report.md             # This file
└── README.md             # Setup and usage instructions
```

**Dependencies:** See `final/draft/README.md` for environment setup.

---

**Total experiment time:** ~*[X]* hours
**Analysis time:** ~*[Y]* hours
**Report writing time:** ~*[Z]* hours
