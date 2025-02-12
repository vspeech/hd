# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from einops import rearrange
import torchaudio.transforms as T

from harmonic import HarmonicSTFT

LRELU_SLOPE = 0.1

def get_2d_padding(
    kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class MDCUnit(nn.Module):
    """
    resnet
    """
    def __init__(self, in_channels, out_channels, dilations, ks=(5, 5), stride=(1, 1), use_bias=True, use_spectral_norm=0):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm == 0 else spectral_norm
        self.dilations = dilations
        model = nn.ModuleDict()
        cur_channels = in_channels
        for i, dilation in enumerate(self.dilations):
            model["layer_%d" % (2 * i + 1)] = norm_f(nn.Conv2d(cur_channels, cur_channels, kernel_size = ks, stride = stride, dilation=(1, dilation), padding=get_2d_padding(ks, (1, dilation))))
        
        model["layer_%d" % (2 * len(self.dilations) + 1)] = nn.Sequential(
            nn.LeakyReLU(0.2),
            norm_f(nn.Conv2d(cur_channels, out_channels, kernel_size = ks, stride = (2, 1), padding = get_2d_padding(ks)))
        )
        self.model = model 

    def forward(self, x):
        """
        forward
        """
        for key, layer in self.model.items():
            x = layer(x)
        return x


class Conv2dNet2(nn.Module):
    """
    conv2d net base
    """
    def __init__(self, in_channels=2, ndf=32, out_channels=[64,64,128], n_layers=3, dilations=[1, 2, 4], n_freqs=513, use_spectral_norm=0):
        super().__init__()
        self.n_layers = n_layers
        self.n_freqs = n_freqs
        self.dilations = dilations
        model = nn.ModuleDict()
        norm_f = weight_norm if use_spectral_norm == 0 else spectral_norm
        self.dp_conv1 = nn.Sequential(
            norm_f(nn.Conv2d(in_channels, in_channels, kernel_size = (7, 7), stride = 1, padding = (3, 3), groups=in_channels, padding_mode = "reflect")),
            norm_f(nn.Conv2d(in_channels, ndf, kernel_size = (1, 1), stride = 1)),
        )
        self.norm_conv1 = norm_f(nn.Conv2d(in_channels, ndf, kernel_size = (7, 7), stride = 1, padding = (3, 3), padding_mode = "reflect"))
        cur_channels = ndf
        for i in range(self.n_layers):
            model["layer_%d" % (2 * i + 1)] = MDCUnit(cur_channels, out_channels[i], dilations = self.dilations, ks = (5, 5), stride = (1, 1), use_spectral_norm=use_spectral_norm)
            cur_channels = out_channels[i]

        #ks = (n_freqs // 4 ** (n_layers)) + 1
        ks = (n_freqs // 2 ** (n_layers)) + 1
        ks = max(3, ks)
        self.conv_post = nn.Sequential(
            nn.LeakyReLU(0.2),
            norm_f(nn.Conv2d(cur_channels, 1, kernel_size = (ks, 1), padding = (ks // 2, 0)))
        )
        self.model = model

    def forward(self, x_in):
        """ x ---  (B, 2, F, T)
        """
        results = []
        fmap = []
        x1 = self.dp_conv1(x_in)
        x2 = self.norm_conv1(x_in)
        x = x1 + x2
        fmap.append(x)
        for key, layer in self.model.items():
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        return x, fmap

class DiscriminatorHDF(nn.Module):
    """
    stft gan
    """
    def __init__(self, ndf=32, out_channels=[32,32,32], n_layers=3, dilations=[1,2,4], fft_size=1024, shift_size=256, win_size=1024, window="hann_window", sample_rate=24000, n_harmonic=10, semitone_scale=2, learn_bw="only_Q", n_freqs=124, use_spectral_norm=0):
        super().__init__()

        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_size = win_size
        self.register_buffer("window", getattr(torch, window)(win_size))

        self.hstft = HarmonicSTFT(sample_rate=sample_rate,
                                    n_fft=fft_size,
                                    win_length=win_size,
                                    hop_length=shift_size,
                                    n_harmonic=n_harmonic,
                                    n_filter=n_freqs,
                                    semitone_scale=semitone_scale,
                                    learn_bw=learn_bw)
        self.model = Conv2dNet2(n_harmonic, ndf, out_channels, n_layers, dilations, n_freqs = n_freqs, use_spectral_norm=use_spectral_norm)

    def forward(self, input):
        """ input --- signal (B, T)
        """
        results = []
        input = input.squeeze(1)
        x = self.hstft(input)
        x, fmap = self.model(x)

        return x, fmap

class HarmonicDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(HarmonicDiscriminator, self).__init__()

        self.cfg = cfg

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorHDF(
                    ndf=cfg.ndf,
                    out_channels=cfg.out_channels,
                    n_layers=cfg.n_layers,
                    dilations=cfg.dilations,
                    fft_size=cfg.n_ffts[0],
                    shift_size=cfg.hop_lengths[0],
                    win_size=cfg.win_lengths[0],
                    window="hann_window",
                    sample_rate=cfg.sampling_rate,
                    n_harmonic=cfg.n_harmonics[0],
                    semitone_scale=cfg.semitone_scale,
                    learn_bw=cfg.learn_bw,
                    n_freqs=cfg.n_freqs[0],
                )
                for i in range(len(cfg.n_freqs))
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
