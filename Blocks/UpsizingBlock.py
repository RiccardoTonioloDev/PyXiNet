from torch import nn
import torch


class UpsizingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpsizingBlock, self).__init__()
        self.__block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
            ),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__block(x)
