import math
from math import sqrt
from torch import nn
import torch
import torch.nn.functional as F

from urllib.request import urlopen
from PIL import Image
import timm


class Sine(nn.Module):
    """Sine activation with scaling.

    Args:

    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.
    Args:
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        is_last=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        if self.is_last:
            out += 0
        else:
            out = self.activation(out)
        return out


class EncoderToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.
    Args:
    """

    def __init__(self, num_modulations, kernel_size, latent_channels):
        super().__init__()
        self.num_modulations = num_modulations
        self.kernel_size = kernel_size
        self.latent_channels = latent_channels


        self.in_ch = 4
        self.w_size = kernel_size
        self.w_half = math.floor(self.w_size / 2)
        self.convEnc = nn.Conv2d(self.in_ch, self.latent_channels, self.w_size, bias=False)
        self.pixshuf = nn.PixelShuffle(2)

        self.maxvit_encoder = timm.create_model(
            'maxvit_rmlp_small_rw_224',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )

        self.out = nn.Linear(768, num_modulations)


    def forward(self, img):
        img_pad = F.pad(img, [self.w_half, self.w_half, self.w_half, self.w_half], "reflect")
        enc_LR = self.convEnc(img_pad)
        initRGB = self.pixshuf(enc_LR)
        max_enc = self.maxvit_encoder(initRGB)
        return self.out(max_enc)


