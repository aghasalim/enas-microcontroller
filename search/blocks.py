"""Gen-1 candidate: shift-mixed, asymmetric-depthwise net for <250KB SRAM MCUs."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerOfTwoReLU(nn.Module):
    """ReLU6 plus a leak scaled by a power of two.

    The leak is x * 2**-shift. In a fixed-point int8 pipeline that constant is
    an arithmetic right shift, not a multiply, so the extra branch costs one
    instruction on a Cortex-M4 and nothing in the weight budget. The clamp at 6
    keeps the output range fixed, which is what lets the whole activation fold
    into the requantisation step.
    """

    def __init__(self, shift: int = 4, ceil: float = 6.0):
        super().__init__()
        self.scale = 2.0 ** -shift
        self.ceil = ceil

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0, max=self.ceil) + self.scale * torch.clamp(x, max=0.0)


class GroupShift(nn.Module):
    """Zero-parameter spatial mixing: roll channel groups in four directions.

    Replaces the spatial half of a convolution with a memory offset. Costs no
    weights, no multiplies, and on the device it is pointer arithmetic. The
    remaining channels are left in place so the block keeps a centre tap.
    """

    def __init__(self, channels: int, amount: int = 1):
        super().__init__()
        self.amount = amount
        g = channels // 5
        # index boundaries for the five groups: up, down, left, right, centre
        self.bounds = (g, 2 * g, 3 * g, 4 * g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c, d = self.bounds
        n = self.amount
        return torch.cat([
            torch.roll(x[:, :a], shifts=-n, dims=2),
            torch.roll(x[:, a:b], shifts=n, dims=2),
            torch.roll(x[:, b:c], shifts=-n, dims=3),
            torch.roll(x[:, c:d], shifts=n, dims=3),
            x[:, d:],
        ], dim=1)


class SparseResidual(nn.Module):
    """Residual that needs no projection when the channel count changes.

    A 1x1 projection on a skip path is pure overhead on this budget. Instead the
    input is strided in the channel dimension to whatever width the block needs,
    which is a fixed gather with no weights. Downsampling in space is average
    pooling for the same reason.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.out_ch = out_ch
        self.stride = stride
        idx = torch.linspace(0, in_ch - 1, steps=out_ch).round().long()
        self.register_buffer("idx", idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = F.avg_pool2d(x, self.stride)
        return x.index_select(1, self.idx)


class ShiftBlock(nn.Module):
    """Shift for spatial mixing, then one pointwise convolution."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, shift: int = 4):
        super().__init__()
        self.shift = GroupShift(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = PowerOfTwoReLU(shift)
        self.res = SparseResidual(in_ch, out_ch, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn(self.pw(self.shift(x))))
        return y + self.res(x)


class AsymBlock(nn.Module):
    """A 5x5 depthwise receptive field bought as 1x5 then 5x1.

    Two strips cost 10 weights per channel against 25 for the square kernel, and
    the intermediate never materialises at full 5x5 width, which matters more
    for peak SRAM than the weight saving does.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, k: int = 5, shift: int = 4):
        super().__init__()
        pad = k // 2
        self.dw_h = nn.Conv2d(in_ch, in_ch, (1, k), padding=(0, pad), groups=in_ch, bias=False)
        self.dw_v = nn.Conv2d(in_ch, in_ch, (k, 1), stride=(stride, stride),
                              padding=(pad, 0), groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.act = PowerOfTwoReLU(shift)
        self.res = SparseResidual(in_ch, out_ch, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn_dw(self.dw_v(self.dw_h(x))))
        y = self.bn_pw(self.pw(y))
        return self.act(y) + self.res(x)


class MicroNetGen1(nn.Module):
    """Gen-1 topology. Widths are multiples of 8 so int8 kernels stay aligned."""

    def __init__(self, num_classes: int = 10, in_ch: int = 3, widths=(24, 48, 96, 128)):
        super().__init__()
        w0, w1, w2, w3 = widths
        # One real convolution, at stride 2, so nothing downstream ever sees
        # full resolution. This is the single biggest lever on peak SRAM.
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w0),
            PowerOfTwoReLU(),
        )
        self.blocks = nn.Sequential(
            ShiftBlock(w0, w1, stride=1),
            AsymBlock(w1, w2, stride=2),
            ShiftBlock(w2, w2, stride=1),
            AsymBlock(w2, w3, stride=2),
            ShiftBlock(w3, w3, stride=1),
        )
        self.head = nn.Linear(w3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

    @torch.no_grad()
    def prune_pointwise(self, keep: float = 0.75) -> int:
        """Structurally zero the lowest L1 pointwise filters, per block.

        Masking here rather than slicing keeps the graph shape fixed so the
        model stays trainable and exportable; the zeroed filters are what a
        later export step physically removes.
        """
        removed = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.out_channels > 8:
                l1 = m.weight.abs().sum(dim=(1, 2, 3))
                n_cut = int(m.out_channels * (1.0 - keep))
                if n_cut:
                    cut = torch.topk(l1, n_cut, largest=False).indices
                    m.weight[cut] = 0.0
                    removed += n_cut
        return removed


if __name__ == "__main__":
    model = MicroNetGen1()
    n = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"params: {n:,}")
    print(f"forward: {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"int8 weight footprint: {n / 1024:.1f} KB")
    model.train()
    loss = y.sum()
    loss.backward()
    print("backward: ok")
    print("pruned filters:", model.prune_pointwise())
