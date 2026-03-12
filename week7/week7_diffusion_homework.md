# DATA 37100 — Week 7 Homework: Diffusion Models

**Due:** (set by instructor)

This homework has two parts:
1) short conceptual questions,
2) a small coding task using the Week 7 diffusion lab code.

---

## Part 1 — Concept (20–30 min)

1) In one paragraph, explain the “forward” diffusion process and what it accomplishes.

2) Why is the DDPM training objective often written as an MSE on predicted noise \(\varepsilon\) rather than a direct likelihood on pixels?

3) Temperature in decoding (Week 6) changes randomness at inference time.  
What is the closest “analogy knob” in diffusion sampling, and what does it trade off?

4) Compare mode collapse in GANs to diversity issues in diffusion.  
Are they the same phenomenon? If not, what’s different?

---

## Part 2 — Coding (60–90 min)

You will train a small diffusion model on MNIST or Fashion-MNIST and answer a few analysis questions.

### A) Run a baseline
From `week7/src`:

```bash
python lab07_diffusion_core.py --dataset mnist --epochs 1 --bs 128 --T 200 --save-every 200
```

This should create an output folder like:
`week7/src/untrack/outputs/lab07/<timestamp>/`

**Deliverable A1:** Include a grid of samples from your trained model (PNG).  
**Deliverable A2:** Include a grid showing intermediate denoising steps (PNG).

### B) Ablations (pick two)
Pick **two** of the following ablations:

- Change steps: \(T\in\{50, 100, 200, 400\}\)
- Change model width: `--base-ch 32` vs `--base-ch 64`
- Change schedule endpoints: `--beta1 1e-4 --beta2 0.02` (baseline) vs a more aggressive schedule
- Train on Fashion-MNIST instead of MNIST

**Deliverable B:** For each ablation, provide:
- one sentence describing what you changed,
- one sample grid,
- one sentence describing what happened (quality/diversity/runtime).

### C) Short analysis
Answer:
1) Which change most improved sample quality? Why do you think so?
2) Which change most harmed diversity? Why do you think so?
3) If you had 10× more compute, what is the *first* improvement you would try (architecture, schedule, training objective, data, etc.)?

---

## Submission
Submit:
- a short PDF (or markdown) report with images + answers
- any code changes (if you modified the lab)
