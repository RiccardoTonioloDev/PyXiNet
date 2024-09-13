import torch
from torch import nn


class LightSelfAttentionCNN1(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int):
        super(LightSelfAttentionCNN1, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.pre_norm = nn.LayerNorm([in_channels, height // 2, width // 2])
        self.post_norm = nn.LayerNorm([in_channels, height // 2, width // 2])

    def forward(self, x):
        batch_size, C, width, height = x.size()
        width, height = width // 2, height // 2
        x = nn.functional.interpolate(x, size=(width, height), mode="area")
        x = self.pre_norm(x)

        # Convoluzioni per ottenere query, key e value
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        # Calcolo dell'attenzione
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)

        # Applicazione dell'attenzione ai valori
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # Connessione residua
        out = self.gamma * out + x
        out = self.post_norm(out)
        out = nn.functional.interpolate(out, size=(width * 2, height * 2), mode="area")
        return out


class LightSelfAttentionCNN2(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int):
        super(LightSelfAttentionCNN2, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.pre_norm = nn.LayerNorm([in_channels, height // 2, width // 2])

    def forward(self, x):
        batch_size, C, width, height = x.size()
        width, height = width // 2, height // 2
        ds_x = nn.functional.interpolate(x, size=(width, height), mode="area")
        ds_x = self.pre_norm(ds_x)

        # Convoluzioni per ottenere query, key e value
        query = (
            self.query_conv(ds_x).view(batch_size, -1, width * height).permute(0, 2, 1)
        )
        key = self.key_conv(ds_x).view(batch_size, -1, width * height)
        value = self.value_conv(ds_x).view(batch_size, -1, width * height)

        # Calcolo dell'attenzione
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)

        # Applicazione dell'attenzione ai valori
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # Connessione residua
        gamma_out = self.gamma * out
        gamma_out = nn.functional.interpolate(
            gamma_out, size=(width * 2, height * 2), mode="area"
        )
        out = gamma_out + x
        return out


class CBAM(nn.Module):
    """
    References:
        - Spatial attention: https://paperswithcode.com/method/spatial-attention-module
        - Channel attention: https://paperswithcode.com/method/channel-attention-module
        - 3D attention: https://joonyoung-cv.github.io/assets/paper/20_ijcv_a_simple.pdf
    """

    def __init__(self, in_channels: int, reduction_ratio=8):
        assert (
            in_channels >= 16
        ), "Input channels have to be greater than 16 for the attention module to work correctly"
        super(CBAM, self).__init__()
        #################################
        #             CHANNEL           #
        #################################
        self.channels_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
        )
        #################################
        #             SPATIAL           #
        #################################
        self.spacial_net = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
        )

    def forward(self, x: torch.Tensor):
        #################################
        #             CHANNEL           #
        #################################
        avg_pool = nn.functional.avg_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )
        max_pool = nn.functional.max_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )
        channel_att: torch.Tensor = self.channels_mlp(avg_pool) + self.channels_mlp(
            max_pool
        )
        channel_att = channel_att.sigmoid().unsqueeze(2).unsqueeze(3)
        residual_channel_att = x * channel_att
        #################################
        #             SPATIAL           #
        #################################
        compressed_x = torch.cat(
            (
                torch.max(residual_channel_att, 1)[0].unsqueeze(1),
                torch.mean(residual_channel_att, 1).unsqueeze(1),
            ),
            dim=1,
        )
        spacial_att: torch.Tensor = self.spacial_net(compressed_x).sigmoid()
        #################################
        #            COMBINED           #
        #################################
        residual_spacial_attention = x * spacial_att
        return residual_spacial_attention + x
