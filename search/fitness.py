"""Train a candidate briefly and score it.

Two things are measured rather than estimated: parameter count, and peak
activation tensor size, which is what actually decides whether a model fits in
SRAM. Weights are the number people quote; activations are the number that
makes the deployment fail.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def peak_activation(model: nn.Module, shape) -> int:
    """Largest single tensor produced by any module, in elements.

    A real arena allocator does better than this by reusing buffers, so treat it
    as an upper bound on the working set rather than a prediction.
    """
    peak = 0

    def hook(_m, _i, out):
        nonlocal peak
        if isinstance(out, torch.Tensor):
            peak = max(peak, out.numel())

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    was_training = model.training
    model.eval()
    model(torch.zeros(1, *shape))
    for h in handles:
        h.remove()
    model.train(was_training)
    return peak


@torch.no_grad()
def macs(model: nn.Module, shape) -> int:
    """Multiply-accumulates, counted from the convolutions actually executed."""
    total = 0

    def hook(m, _i, out):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            total += out.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    was_training = model.training
    model.eval()
    model(torch.zeros(1, *shape))
    for h in handles:
        h.remove()
    model.train(was_training)
    return total


def evaluate(model, loader, device="cpu") -> float:
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1)
            correct += (pred == y.to(device)).sum().item()
            n += y.numel()
    return correct / max(n, 1)


def train_micro(model, train_loader, val_loader, epochs: int, lr: float = 3e-3,
                device: str = "cpu") -> tuple[float, float]:
    """The micro-epoch phase. Short on purpose: the search needs a ranking
    signal, not a final number, and a short budget is what makes the search
    affordable. The winner is retrained properly afterwards."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    steps = max(1, epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    t0 = time.perf_counter()
    for _ in range(epochs):
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return evaluate(model, val_loader, device), time.perf_counter() - t0


def fitness(acc: float, params: int, macs_count: int, peak_act: int,
            alpha: float = 1.0, beta: float = 0.02, gamma: float = 0.02,
            param_cap: int = 50_000, sram_cap_bytes: int = 250 * 1024) -> float:
    """F = alpha*Acc - beta*log10(params) - gamma*log10(MACs), with hard caps.

    The caps are not soft penalties. A model over 50k parameters, or whose
    weights plus peak activation will not fit in 250KB at int8, cannot be
    deployed at all, so it scores negative infinity rather than a good accuracy
    with a small deduction. A fitness function that lets an undeployable model
    win is measuring the wrong thing.
    """
    if params > param_cap:
        return float("-inf")
    if params + peak_act > sram_cap_bytes:      # int8: one byte per weight, per activation
        return float("-inf")
    return alpha * acc - beta * math.log10(max(params, 1)) - gamma * math.log10(max(macs_count, 1))
