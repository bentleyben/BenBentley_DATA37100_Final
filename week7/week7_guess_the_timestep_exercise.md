# Week 7 — Diffusion Models  
## In-Class Exercise: "Guess the Timestep"

### Goal
Build intuition for how noise increases across diffusion timesteps and how the model learns to reverse the process.

---

## Part 1 — Observation (5 min)

You will be shown several noisy images sampled from different diffusion timesteps (in folder week7/media/timesteps).

For each image:

1. Estimate whether the timestep is:
   - ☐ Early (almost clean)
   - ☐ Middle (partially corrupted)
   - ☐ Late (almost pure noise)

2. Briefly describe the visual clues you used:
```
________________________________________________________
________________________________________________________
```

---

## Part 2 — Pair Discussion (5 min)

Discuss with a partner:

- Which visual features disappear first as noise increases?
- At what stage does the digit become difficult to recognize?
- Why might later denoising steps be harder for the model?

Write one key observation:
```
________________________________________________________
```

---

## Part 3 — Reverse Thinking (5 min)

Consider the reverse process:

1. If you start from pure noise, which steps are likely:
   - Harder to predict?
   - Easier to predict?

Explain briefly:
```
________________________________________________________
________________________________________________________
```

---

## Reflection Question

Why does diffusion training use **noise prediction** instead of directly predicting the clean image?

Write one sentence:
```
________________________________________________________
```
