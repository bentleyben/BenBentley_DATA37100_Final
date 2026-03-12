# DATA 37100 — Week 7 In-Class Worksheet: Diffusion Models (DDPM)

**Goal:** Build intuition for the forward noising process, the reverse denoising process, and why the “predict noise” objective is so effective.

---

## Part A — Forward diffusion as “information destruction”

We define a Markov chain that gradually adds Gaussian noise:

\[
q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\; \beta_t I),
\quad t=1,2,\dots,T
\]

1) In words, what does \(\beta_t\) control?

> ______________________________________________________________________

2) If \(\beta_t\) is too large early in the chain, what happens to the difficulty of the reverse process?

> ______________________________________________________________________

3) Define \(\alpha_t = 1-\beta_t\) and \(\bar\alpha_t = \prod_{s=1}^t \alpha_s\).  
Show the closed form:

\[
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,I)
\]

> Sketch derivation idea (you don’t need every algebra step):
>
> ______________________________________________________________________

---

## Part B — Reverse diffusion as “learned denoising”

We want a reverse-time chain:

\[
p_\theta(x_{t-1}\mid x_t)
\]

A standard DDPM parameterization predicts **noise**:

\[
\varepsilon_\theta(x_t,t) \approx \varepsilon
\]

4) Why might predicting \(\varepsilon\) be easier/more stable than predicting \(x_0\) directly?

> ______________________________________________________________________

5) If the model predicts \(\varepsilon_\theta\), we can estimate \(x_0\) via:

\[
\hat x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\Big(x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta(x_t,t)\Big)
\]

What happens to this formula as \(t\to T\) (very noisy)? Why is that a hint that we should be careful with schedules?

> ______________________________________________________________________

---

## Part C — What you will see in the lab

In the lab, you will:
- train a small CNN \(\varepsilon_\theta\) on MNIST (or Fashion-MNIST),
- sample by starting from pure noise \(x_T\sim\mathcal{N}(0,I)\),
- denoise step-by-step to produce images.

6) During sampling, you will see intermediate images at different \(t\).  
What qualitative “phase transition” do you expect as \(t\) decreases?

> ______________________________________________________________________

7) **Mini-experiment:** keep the network fixed, and only change:
- number of diffusion steps \(T\),
- schedule shape (linear vs “cosine” — optional extension),
- classifier-free guidance strength (optional extension).

For each knob, predict the effect on:
- sample quality,
- diversity,
- runtime.

> ______________________________________________________________________

---

## Part D — Failure modes (tie back to Weeks 3–6)

8) Name two failure modes you might encounter with diffusion training in this small lab setup, and a likely cause for each.

| Failure mode | Symptom | Likely cause | Quick fix |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |

---

### Turn-in (in class)
Upload a photo/screenshot of your completed worksheet (or type answers) to Canvas (if required by instructor).
