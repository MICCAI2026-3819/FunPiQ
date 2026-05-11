import torch
import torch.nn as nn

from badish.model.space_admiral import SpaceAdmiral
from badish.utils.registry import ModelRegistry


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels for NCHW tensors."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, act="gelu", drop_path=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = ChannelLayerNorm(dim)
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.act = self._make_activation(act)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.drop_path = DropPath(drop_path)

    @staticmethod
    def _make_activation(name):
        if name == "gelu":
            return nn.GELU()
        if name == "silu":
            return nn.SiLU()
        if name == "lrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise ValueError("act must be one of: gelu | silu | lrelu")

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.gamma * x
        return residual + self.drop_path(x)


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = ChannelLayerNorm(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(self.norm(x))


class ModernHead64to16(nn.Module):
    def __init__(self, in_ch=1024, mid_ch=512, mlp_ratio=4, act="gelu", drop_path=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            ChannelLayerNorm(in_ch),
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=True),
        )
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(mid_ch, mlp_ratio, act, drop_path=drop_path * 0.33),
            ConvNeXtBlock(mid_ch, mlp_ratio, act, drop_path=drop_path * 0.33),
        )
        self.down1 = Downsample(mid_ch, mid_ch)
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(mid_ch, mlp_ratio, act, drop_path=drop_path * 0.66),
            ConvNeXtBlock(mid_ch, mlp_ratio, act, drop_path=drop_path * 0.66),
        )
        self.down2 = Downsample(mid_ch, mid_ch)
        self.stage3 = ConvNeXtBlock(mid_ch, mlp_ratio, act, drop_path=drop_path)
        self.head = nn.Sequential(
            ChannelLayerNorm(mid_ch),
            nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        return self.head(x)


@ModelRegistry.register("sa_cnn")
class SpaceAdmiralCNN(SpaceAdmiral):
    def __init__(
        self,
        in_channels=1024,
        out_channels=1,
        mid_channels=512,
        mlp_ratio=4,
        act="gelu",
        drop_path=0.1,
    ):
        super().__init__()
        if out_channels != 1:
            raise ValueError("SpaceAdmiralCNN only supports out_channels=1")
        self.model = ModernHead64to16(
            in_ch=in_channels,
            mid_ch=mid_channels,
            mlp_ratio=mlp_ratio,
            act=act,
            drop_path=drop_path,
        )
