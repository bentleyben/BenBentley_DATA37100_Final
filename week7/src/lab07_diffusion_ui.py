#!/usr/bin/env python3
#%%
"""
DATA 37100 — Week 7 Lab: Diffusion Models (DDPM-lite)
Interactive UI (ipywidgets) for training + sampling.

Run (VSCode Interactive / Jupyter):
- Open this file and run cells.

This UI wraps the core functions in lab07_diffusion_core.py.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import matplotlib.pyplot as plt

from lab07_diffusion_core import (
    DiffusionSchedule,
    TinyEpsModel,
    get_device,
    get_dataloader,
    make_linear_schedule,
    q_sample,
    sample,
    seed_all,
)

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
except Exception as e:  # pragma: no cover
    raise RuntimeError("ipywidgets is required for the UI: pip install ipywidgets") from e


def show_grid(x: torch.Tensor, title: str = "", nrow: int | None = None) -> None:
    x = x.detach().cpu()
    x01 = (x + 1) * 0.5
    n = x01.size(0)
    if nrow is None:
        nrow = int(math.sqrt(n))
    # simple grid without torchvision dependency here (matplotlib)
    cols = nrow
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(cols * 1.2, rows * 1.2))
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(x01[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.show()


def build_ui() -> None:
    # state
    state: Dict[str, object] = {
        "device": None,
        "model": None,
        "sch": None,
        "dl": None,
        "opt": None,
        "step": 0,
    }

    # controls
    dataset = widgets.Dropdown(options=[("MNIST", "mnist"), ("Fashion-MNIST", "fashion")], value="mnist", description="Dataset")
    device_dd = widgets.Dropdown(options=[("auto", "auto"), ("mps", "mps"), ("cuda", "cuda"), ("cpu", "cpu")], value="auto", description="Device")
    seed = widgets.IntText(value=42, description="Seed")

    T = widgets.IntSlider(value=200, min=50, max=500, step=50, description="T")
    beta1 = widgets.FloatText(value=1e-4, description="beta1")
    beta2 = widgets.FloatText(value=0.02, description="beta2")

    base_ch = widgets.IntSlider(value=64, min=32, max=128, step=32, description="base_ch")
    time_dim = widgets.IntSlider(value=128, min=64, max=256, step=64, description="t_dim")
    lr = widgets.FloatText(value=2e-4, description="lr")
    bs = widgets.Dropdown(options=[64, 128, 256], value=128, description="batch")
    iters = widgets.IntText(value=400, description="train iters")

    target = widgets.Dropdown(options=[("eps (predict noise)", "eps"), ("x0 (predict clean image)", "x0")], value="eps", description="target")

    sample_n = widgets.Dropdown(options=[16, 36, 64, 100], value=64, description="sample_n")

    btn_init = widgets.Button(description="Init", button_style="info")
    btn_train = widgets.Button(description="Train", button_style="success")
    btn_sample = widgets.Button(description="Sample", button_style="warning")

    out = widgets.Output()
    # --- layout tuning (UI ergonomics) ---
    # Keep control widths consistent so rows don't sprawl horizontally.
    # (Works well in VSCode notebooks and classic Jupyter.)
    W_DD = "210px"      # dropdown/text controls
    W_NUM = "170px"     # small numeric fields
    W_SL = "320px"      # sliders
    DESC = "95px"       # label column width

    def stylize(w, width=None):
        if width is not None:
            w.layout.width = width
        w.style.description_width = DESC
        return w

    # apply widths
    stylize(dataset, W_DD)
    stylize(device_dd, W_DD)
    stylize(seed, W_NUM)

    stylize(T, W_SL)
    stylize(beta1, W_NUM)
    stylize(beta2, W_NUM)

    stylize(base_ch, W_SL)
    stylize(time_dim, W_SL)

    stylize(lr, W_NUM)
    stylize(bs, W_DD)

    stylize(iters, W_NUM)
    stylize(target, "260px")
    stylize(sample_n, W_DD)

    # button sizing
    for b in (btn_init, btn_train, btn_sample):
        b.layout.width = "110px"
        b.layout.height = "36px"


    def do_init(_=None):
        with out:
            clear_output(wait=True)
            seed_all(seed.value)
            dev = get_device(device_dd.value)
            print(f"[Device] {dev}")
            sch = make_linear_schedule(T.value, beta1.value, beta2.value, device=dev)
            dl = get_dataloader(dataset.value, "../../data/bigdata/MNIST", bs.value, num_workers=0)
            model = TinyEpsModel(base_ch=base_ch.value, time_emb_dim=time_dim.value).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=lr.value)
            state.update({"device": dev, "sch": sch, "dl": dl, "model": model, "opt": opt, "step": 0})
            print(f"[Init] ready. target={target.value}")

    def do_train(_=None):
        if state["model"] is None:
            do_init()
        model: TinyEpsModel = state["model"]  # type: ignore
        sch: DiffusionSchedule = state["sch"]  # type: ignore
        dl = state["dl"]  # type: ignore
        opt = state["opt"]  # type: ignore
        dev: torch.device = state["device"]  # type: ignore

        model.train()
        step = int(state["step"])
        t0 = time.time()
        with out:
            clear_output(wait=True)
            print(f"[Train] iters={iters.value} starting at step={step} ...")
            it = iter(dl)
            for k in range(iters.value):
                try:
                    x0, _ = next(it)
                except StopIteration:
                    it = iter(dl)
                    x0, _ = next(it)
                x0 = x0.to(dev)
                B = x0.size(0)
                t = torch.randint(0, sch.T, (B,), device=dev, dtype=torch.long)
                eps = torch.randn_like(x0)
                x_t = q_sample(x0, t, eps, sch)
                pred = model(x_t, t)
                if target.value == "eps":
                    loss = torch.mean((pred - eps) ** 2)
                else:
                    # Predict x0 directly
                    loss = torch.mean((pred - x0) ** 2)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if (k + 1) % 50 == 0:
                    print(f" step {step+k+1:06d} | loss={loss.item():.4f}")
            step += iters.value
            state["step"] = step
            dt = time.time() - t0
            print(f"[Train] done in {dt:.1f}s. step={step}")

    def do_sample(_=None):
        if state["model"] is None:
            do_init()
        model: TinyEpsModel = state["model"]  # type: ignore
        sch: DiffusionSchedule = state["sch"]  # type: ignore
        dev: torch.device = state["device"]  # type: ignore

        with out:
            clear_output(wait=True)
            print("[Sample] generating ...")
            xs, inter = sample(model, sch, n=sample_n.value, device=dev, intermediate_ts=(sch.T-1, int(sch.T*0.75), int(sch.T*0.5), int(sch.T*0.25), 0), target=target.value)
            show_grid(xs, title="Final samples")
            # show intermediates (a few)
            for t in sorted(inter.keys(), reverse=True):
                show_grid(inter[t], title=f"Intermediate t={t}")

    btn_init.on_click(do_init)
    btn_train.on_click(do_train)
    btn_sample.on_click(do_sample)

    controls1 = widgets.HBox([dataset, device_dd, seed], layout=widgets.Layout(flex_flow="row wrap", gap="12px"))
    controls2 = widgets.HBox([T, beta1, beta2], layout=widgets.Layout(flex_flow="row wrap", gap="12px"))
    controls3 = widgets.HBox([base_ch, time_dim], layout=widgets.Layout(flex_flow="row wrap", gap="12px"))
    controls4 = widgets.HBox([lr, bs], layout=widgets.Layout(flex_flow="row wrap", gap="12px"))
    controls5 = widgets.HBox([iters, target, sample_n], layout=widgets.Layout(flex_flow="row wrap", gap="12px"))
    buttons = widgets.HBox([btn_init, btn_train, btn_sample], layout=widgets.Layout(flex_flow="row wrap", gap="10px"))

    display(widgets.VBox([controls1, controls2, controls3, controls4, controls5, buttons, out], layout=widgets.Layout(gap="10px")))


if __name__ == "__main__":
    build_ui()

#%%
