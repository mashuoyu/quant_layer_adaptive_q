# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,

# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any, Callable, List, Optional, Tuple, Union
#from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.ans import BufferedRansEncoder, RansDecoder
from entropy_models import EntropyBottleneck, GaussianConditional
from layers import GDN, MaskedConv2d,ResidualBlockUpsample, ResidualBlock, conv3x3,conv1x1,gumbel_softmax_STE_test
from base import (

    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from utils import conv, deconv,depthconv

__all__ = [
    "CompressionModel",
    "FactorizedPrior",
    "FactorizedPriorReLU",
    "ScaleHyperprior",
    "MeanScaleHyperprior",
    "JointAutoregressiveHierarchicalPriors",
    "get_scale_table",
    "SCALES_MIN",
    "SCALES_MAX",
    "SCALES_LEVELS",
]



class FactorizedPrior(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y
            x ──►─┤g_a├──►─┐
                  └───┘    │
                           ▼
                         ┌─┴─┐
                         │ Q │
                         └─┬─┘
                           │
                     y_hat ▼
                           │
                           ·
                        EB :
                           ·
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(M)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2**4

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict,stricr=False)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}



class FactorizedPriorReLU(FactorizedPrior):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.
    GDN activations are replaced by ReLU.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, 3),
        )



class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

        self.qy_param = torch.nn.Parameter(torch.zeros(M,1,1))
        self.qz_param = torch.nn.Parameter(torch.ones(N,1,1))


        self.get_q=nn.Sequential(
            conv(1,4,stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(4,12,stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(12,3),
            nn.ReLU(inplace=True),
            conv(3,3),
            nn.AdaptiveAvgPool2d((1,1)),
        )

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict,strict=False)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    

        
    
    def quant_res(self, inputs: Tensor, qp: Tensor, mode: str, means: Optional[Tensor] = None):
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs_soft=inputs/qp + noise
            return inputs_soft

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        
        outputs = torch.round(outputs/qp)

        if mode == "dequantize":
            if means is not None:
                outputs += means/qp
            return outputs
        
        
        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs


    def update(self, scale_table=None, force=False):
        """Updates EntropyBottleneck and GaussianConditional CDFs.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (torch.Tensor): table of scales (i.e. stdev)
                for initializing the Gaussian distributions
                (default: 64 logarithmically spaced scales from 0.11 to 256)
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if at least one of the modules was updated.
        """
        qz=self.qz_param
        qy=self.qy_param
        if scale_table is None:
            scale_table = get_scale_table()
        updated = False
        for _, module in self.named_modules():
            if isinstance(module, EntropyBottleneck):
                updated |= module.update(qz,force=force)
            if isinstance(module, GaussianConditional):
                updated |= module.update_scale_table(scale_table, qy, force=force)
        return updated



class MeanScaleHyperprior(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │                 means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        self.lst = nn.Sequential(
            ResidualBlock(M,M),
            ResidualBlockUpsample(M, M*4//3,2),
            ResidualBlock(M*4//3,M*4//3),
            ResidualBlockUpsample(M*4//3, M*4//3,1),
            ResidualBlock(M*4//3,M*4//3),
            ResidualBlockUpsample(M*4//3, M*4//3,1),
            ResidualBlock(M*4//3,M*4//3),
            ResidualBlockUpsample(M*4//3, M*4//3,1),
            conv3x3(M*4//3,M*4//3),
            nn.LeakyReLU(inplace=True),
        )


    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        N,C,W,H=y.size()
        y_q=torch.zeros((N,C,3,1)).to("cuda")
        for i in range(C):
            y_q_i=y[:,i,:,:].unsqueeze(1)
            y_q_i_after=1+self.get_q(20*y_q_i)
            y_q[:,i,:,:]=y_q_i_after[:,:,:,0]
        qy_average=torch.zeros(N,C,1).to("cuda")
        hard=False
        if self.training:
            qy_soft=F.gumbel_softmax(y_q.squeeze(3), tau=0.03, hard=hard)
            for i in range(3):
                qy_average+=(1+i)*qy_soft[:,:,i].unsqueeze(2)
            qy= qy_average.unsqueeze(3)
        if not self.training:
            qy=1+gumbel_softmax_STE_test(y_q)
        
        qz=torch.ones(128,1,1) 
        qz=qz.to("cuda")
        qy2=torch.ones(192,1,1) 
        qy2=qy2.to("cuda")
        z_hat, z_likelihoods = self.entropy_bottleneck(z,qz)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y1_hat, y1_likelihoods = self.gaussian_conditional(y, qy, scales_hat, means=means_hat)
        y2_hat, y2_likelihoods = self.gaussian_conditional(y, qy2, scales_hat, means=means_hat)

        x_hat = self.g_s(y2_hat)
        f_hat=self.lst(y1_hat)
        if self.training:
            qy_f=(qy_soft.sum(dim=0)).sum(dim=0)/x.size(0)/192
        if not self.training:
            qy_f=torch.histc(qy[:,:,:,:],bins=3,min=1,max=3)/x.size(0)/192
        return {
            "x_hat": x_hat,
            "f_hat": f_hat,
            "q1_f":qy_f,
            "y1_likelihoods":  y1_likelihoods,
            "y2_likelihoods":  y2_likelihoods,
            "z_likelihoods": z_likelihoods,
            
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        qy=self.qy_param
        qz=torch.ones(192,1,1).to("cuda")
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        z_strings = self.entropy_bottleneck.compress(z,qz)
        z_hat = self.entropy_bottleneck.decompress(z_strings, qz, z.size()[-2:])

        z_hat1, z_likelihoods = self.entropy_bottleneck(z,qz)
        bpp_z0=torch.log(z_likelihoods).sum() / (-math.log(2) * num_pixels)
        bpp_z1=0
        for s_i in z_strings:
            bpp_z1 += len(s_i) * 8.0 / num_pixels


        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, qy, indexes, means=means_hat)


        y_hat1, y_likelihoods = self.gaussian_conditional(y, qy, scales_hat, means=means_hat)
        bpp_y0=torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)
        bpp_y1=0
        for s_i in y_strings:
            bpp_y1 += len(s_i) * 8.0 / num_pixels


        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        qy=self.qy_param
        qy=qy.to("cuda")
        qz=torch.ones(192,1,1).to("cuda")
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], qz, shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], qy, indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    
    def get_compression(self,x, training=bool):
        y = self.g_a(x)
        z = self.h_a(y)
        qz=torch.ones(128,1,1) 
        qz=qz.to("cuda")
        qy2=torch.ones(192,1,1) 
        qy2=qy2.to("cuda")
        z_hat, z_likelihoods = self.entropy_bottleneck(z,qz,training)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        return {
            "y_hat":y,
            "scales_hat":scales_hat,
            "means_hat":means_hat,           
        }



class JointAutoregressiveHierarchicalPriors(MeanScaleHyperprior):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                   params ▼
                         └─┬─┘                                          │
                     y_hat ▼                  ┌─────┐                   │
                           ├──────────►───────┤  CP ├────────►──────────┤
                           │                  └─────┘                   │
                           ▼                                            ▼
                           │                                            │
                           ·                  ┌─────┐                   │
                        GC : ◄────────◄───────┤  EP ├────────◄──────────┘
                           ·     scales_hat   └─────┘
                           │      means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional
        EP = Entropy parameters network
        CP = Context prediction (masked convolution)

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict,strict=False)
        return net

    def get_compression_vid(self,x, training=bool):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z,1)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y,1, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        return {
            "y_hat":y,
            "scales_hat":scales_hat,         
            "means_hat":means_hat,
        }
    
    def get_compression_tip(self,x, training=bool):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z,1)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y,1, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y,1, scales_hat, means=means_hat)
        N,C,H,W=y_likelihoods.size()
        y1_likelihoods=y_likelihoods[:,:2*C//3,:,:]
        return {
            "y_hat":y_hat,
            "y1_likelihoods":y1_likelihoods, 
            "y_likelihoods":y_likelihoods,         
        }
    
    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv



class Towards_Scalnable(nn.Module):
    def __init__(self, M, **kwargs):    
        super().__init__(**kwargs)

        self.lst = nn.Sequential(
            ResidualBlock(2*M//3,2*M//3),
            ResidualBlockUpsample(2*M//3, M*4//3,2),
            ResidualBlock(M*4//3,M*4//3),
            ResidualBlockUpsample(M*4//3, M*4//3,1),
            ResidualBlock(M*4//3,M*4//3),
            ResidualBlockUpsample(M*4//3, M*4//3,1),
            ResidualBlock(M*4//3,M*4//3),
            ResidualBlockUpsample(M*4//3, M*4//3,1),
  
            conv3x3(M*4//3,M*4//3),
            #nn.LeakyReLU(inplace=True),
        )
        self.gaussian_conditional = GaussianConditional(None)


    def forward(self, y, y1_likelihoods):
        N,C,W,H=y.size()
        y1=y[:,:2*C//3,:,:]
        f_hat=self.lst(y1)
        return {

            "f_hat": f_hat,
            "y1_likelihoods":y1_likelihoods
        }


class Adaptive_q_embedded(nn.Module):
    def __init__(self, M, **kwargs):    
        super().__init__(**kwargs)

        self.get_q_group=nn.Sequential(
            depthconv(M,3*M,stride=1, kernel_size=3,group=M),
            nn.ReLU(inplace=True),
            depthconv(3*M,3*M,stride=1, kernel_size=3,group=3*M),
            nn.ReLU(inplace=True),
            depthconv(3*M,3*M,group=3*M),
            nn.ReLU(inplace=True),
            conv(3*M,3*M),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.lst = nn.Sequential(
            ResidualBlock(M,M),
            ResidualBlockUpsample(M, M*4//3,2),
            ResidualBlock(M*4//3,M*4//3),
            ResidualBlockUpsample(M*4//3, M*4//3,1),
            ResidualBlock(M*4//3,M*4//3),
            ResidualBlockUpsample(M*4//3, M*4//3,1),
            ResidualBlock(M*4//3,M*4//3),
            ResidualBlockUpsample(M*4//3, M*4//3,1),
            ResidualBlock(M*4//3,M*4//3),
            ResidualBlockUpsample(M*4//3, M*4//3,1),
            conv3x3(M*4//3,M*4//3),
            #nn.LeakyReLU(inplace=True),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.log_scale = torch.nn.Parameter( np.log(np.exp(10**(-5))-1.0) * torch.ones(M) )
        self.eps = 10**(-5)
        self.VID=nn.Sequential(
            conv1x1(M*4//3,2*M),
            nn.ReLU(inplace=True),
            conv1x1(2*M,2*M),
            nn.ReLU(inplace=True),
            conv1x1(2*M,M*4//3),

        )

    def seperat_soft(self,input_latent3):
        N,C,W,H=input_latent3.size()
        num_M=C//3
        y_q=torch.zeros((N,num_M,3,1)).to("cuda")
        for i in range(num_M):
            for j in range(3):
                y_q[:,i,j,0]=input_latent3[:,j+i*3,0,0]
        return y_q

    def forward(self, y,scales_hat, means_hat):
        scale=torch.log(1+torch.exp(self.log_scale))+self.eps
        N,C,W,H=y.size()
        y_q_soft=self.get_q_group(y)
        y_q=self.seperat_soft(y_q_soft)
        qy_average=torch.zeros(N,C,1).to("cuda")
        hard=True
        if self.training:
            qy_soft=F.gumbel_softmax(y_q.squeeze(3), tau=0.03, hard=hard)
            for i in range(3):
                qy_average+=(1+i)*qy_soft[:,:,i].unsqueeze(2)
            qy= qy_average.unsqueeze(3)
        if not self.training:
            qy=1+gumbel_softmax_STE_test(y_q)
        

        y1_hat, y1_likelihoods = self.gaussian_conditional(y, qy, scales_hat, means=means_hat) 

        f_hat=self.lst(y1_hat)
        mu_hat=self.VID(f_hat)
        if self.training:
            qy_f=(qy_soft.sum(dim=0)).sum(dim=0)/y.size(0)/192
        if not self.training:
            qy_f=torch.histc(qy[:,:,:,:],bins=3,min=1,max=3)/y.size(0)/192
        return {

            "f_hat": f_hat,
            "mu_hat":mu_hat,
            "q1_f":qy_f,
            "y1_likelihoods":  y1_likelihoods,
            "scale":scale,
            
        }


