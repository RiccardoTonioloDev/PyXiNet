import torch.nn as nn


def xavier_init(m: nn.Conv2d | nn.ConvTranspose2d):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.trunc_normal_(m.bias)
