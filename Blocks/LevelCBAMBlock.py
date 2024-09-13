from torch import nn
from .LightAttention import CBAM
import torch


class LevelCBAMBlock(nn.Module):
    def __init__(self, in_channels: int, dilation: int = 1):
        super(LevelCBAMBlock, self).__init__()
        padding = dilation

        self.__block = nn.Sequential(
            CBAM(in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            nn.LeakyReLU(0.2),
            CBAM(96),
            nn.Conv2d(
                in_channels=96,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            nn.LeakyReLU(0.2),
            CBAM(64),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            nn.LeakyReLU(0.2),
            CBAM(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__block(x)
