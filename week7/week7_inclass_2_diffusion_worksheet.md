# DATA 37100 — Week 7 In-Class Worksheet
## Diffusion Models: ε-Prediction, SNR, and Sampling

**Name(s): ____________________________**  
**Date: ________________________________**

---

## Part A — What Did the Model Actually Learn?

During training, the model sees pairs (x_t, t).

### Q1
(a) What is the exact regression target in the ε-objective?  

> ____________________________________________________________________

(b) Why is that target distribution stable across timesteps?  

> ____________________________________________________________________

(c) Why does this improve optimization stability?  

> ____________________________________________________________________

---

## Part B — Objective Comparison (x₀ vs ε)

Suppose instead we train with:

L = || x_0 - x_θ(x_t, t) ||²

### Q2
(a) At large timesteps (low SNR), what information about x_0 remains in x_t?  

> ____________________________________________________________________

(b) Why does this make the regression target harder to predict?  

> ____________________________________________________________________

(c) Why does ε-prediction avoid this issue?  

> ____________________________________________________________________

---

## Part C — Schedule and SNR

Recall:

SNR_t = ᾱ_t / (1 - ᾱ_t)

During the stress test (larger beta₂), sample quality degraded.

### Q3
(a) What happened to the SNR curve when beta₂ increased?  

> ____________________________________________________________________

(b) Why did sample quality degrade?  

> ____________________________________________________________________

(c) In one sentence, describe the schedule as a “learning curriculum.”  

> ____________________________________________________________________

---

## Part D — Sampling Reflection (Optional)

### Q4
Why is diffusion sampling slower than GAN sampling?  

> ____________________________________________________________________

---

End of worksheet
