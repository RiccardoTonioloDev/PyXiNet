from torch import nn
import torch
from .LightAttention import CBAM


class DownsizingCBAMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super(DownsizingCBAMBlock, self).__init__()

        self.__block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            CBAM(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            CBAM(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__block(x)
