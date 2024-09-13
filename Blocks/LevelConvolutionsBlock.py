from torch import nn
import torch


class LevelConvolutionsBlock(nn.Module):
    def __init__(self, in_channels: int, dilation: int = 1):
        super(LevelConvolutionsBlock, self).__init__()
        padding = dilation

        self.__block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=96,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            nn.LeakyReLU(0.2),
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
