from typing import List
from torch import nn
from .xinet import XiNet
import torch


class XiEncoder(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        alpha: float,
        gamma: float,
        num_layers: int,
        mid_channels: int,
        out_channels: int,
        with_upscaling: bool = True,
    ):
        super(XiEncoder, self).__init__()

        memo_output_channels = []
        self.xn = XiNet(
            input_shape=input_shape,
            memo_output_channels=memo_output_channels,
            alpha=alpha,
            gamma=gamma,
            num_layers=num_layers,
            base_filters=mid_channels,
        )
        self.out_channels = out_channels
        self.with_upscaling = with_upscaling
        if self.with_upscaling:
            self.deconv = nn.ConvTranspose2d(
                in_channels=memo_output_channels[0],
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
            )
        elif out_channels > 0:
            self.pw_compression = nn.Conv2d(
                in_channels=memo_output_channels[0],
                out_channels=out_channels,
                kernel_size=1,
            )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.xn(x)
        if self.with_upscaling:
            x = self.deconv(x)
        elif self.out_channels > 0:
            x = self.pw_compression(x)
        x = self.activation(x)
        return x
