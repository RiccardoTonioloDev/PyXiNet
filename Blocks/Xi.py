import torch.nn as nn
import torch
import math


# Wrong
def channel_calculator(
    out_channels_first_block: int, alpha: float, beta: float, D_i: int, i: int, N: int
):
    return 4 * math.ceil(
        alpha
        * (2 ** (D_i - 2))
        * (1 + (((beta - 1) * i) / N))
        * out_channels_first_block
    )


class XiConv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gamma: int = 4,
    ):
        super(XiConv2D, self).__init__()

        # Choosing a compression that won't result in a 0 channels output
        gamma = min(out_channels, gamma)

        # Pointwise convolution to reduce dimensionality
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels // gamma, kernel_size=1, stride=1, padding=0
        )

        # Main convolution
        self.conv = nn.Conv2d(
            out_channels // gamma,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.relu6 = nn.ReLU6()

        # ASSUNZIONE: non può essere più efficiente di un solo blocco convoluzionale ammenochè non ci sia qualche sorta
        # di compressione come si fa all'inizio del blocco XiNet. Come di seguito:
        self.attention_module = nn.Sequential(
            torch.nn.Conv2d(
                out_channels, out_channels // gamma, 1, stride=1, padding=0
            ),
            torch.nn.Conv2d(
                out_channels // gamma, out_channels, 3, stride=1, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pointwise_conv(x)
        # compression pointwise conv

        out = self.conv(out)
        # applying main convolution

        out = self.relu6(out)
        # function activation after main convolution

        out = self.attention_module(out)
        # applying attention module

        return out
