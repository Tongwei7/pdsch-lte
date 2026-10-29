"""Depthwise-separable U-Net with RE and PRB/subframe logits."""
import torch
from torch import nn
from torch.nn import functional as F


class Separable(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.GroupNorm(4, cout), nn.ReLU(inplace=True))


class Block(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(Separable(cin, cout), Separable(cout, cout))


class PDSCHUNet(nn.Module):
    def __init__(self, in_channels=3, base=8):
        super().__init__()
        if base < 4 or base % 4:
            raise ValueError("base must be a positive multiple of four")
        self.in_channels, self.base = in_channels, base
        self.e0 = Block(in_channels, base)
        self.e1 = Block(base, base * 2)
        self.e2 = Block(base * 2, base * 4)
        self.bottom = Block(base * 4, base * 8)
        self.d2 = Block(base * 12, base * 4)
        self.d1 = Block(base * 6, base * 2)
        self.d0 = Block(base * 3, base)
        self.re_head = nn.Conv2d(base, 1, 1)
        self.prb_head = nn.Conv2d(base, 1, 1)

    @staticmethod
    def merge(deep: torch.Tensor, skip: torch.Tensor):
        up = F.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((up, skip), dim=1)

    def forward(self, x: torch.Tensor, symbols_per_subframe: int = 14):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Expected [batch, configured channels, subcarrier, symbol]")
        k, t = x.shape[-2:]
        if k % 12 or symbols_per_subframe not in (12, 14) or t % symbols_per_subframe:
            raise ValueError("Input must contain whole LTE PRBs and whole subframes")
        # Padding preserves coordinates; it is removed before either output head.
        x = F.pad(x, (0, (-t) % 8, 0, (-k) % 8))
        a = self.e0(x)
        b = self.e1(F.avg_pool2d(a, 2))
        c = self.e2(F.avg_pool2d(b, 2))
        z = self.bottom(F.avg_pool2d(c, 2))
        z = self.d2(self.merge(z, c))
        z = self.d1(self.merge(z, b))
        z = self.d0(self.merge(z, a))[..., :k, :t]
        pooled = F.avg_pool2d(z, (12, symbols_per_subframe), (12, symbols_per_subframe))
        return {"re": self.re_head(z), "prb": self.prb_head(pooled)}

    def configuration(self):
        return {"in_channels": self.in_channels, "base": self.base}
