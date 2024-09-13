from torch import nn
import torch


class DownsizingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super(DownsizingBlock, self).__init__()

        self.__block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__block(x)
