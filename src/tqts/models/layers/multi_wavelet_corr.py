#!/usr/bin/env python
# coding: utf-8

"""Multi Wavelet Transform layers for TQTS."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

import math
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tqts.models.layers.utils import get_filter


def complex_mul1d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Perform complex multiplication in 1D.

    Args:
        x (torch.Tensor): The input tensor.
        weights (torch.Tensor): The weight tensor.

    Returns:
        torch.Tensor: The result of the complex multiplication.
    """
    return torch.einsum("bix,iox->box", x, weights)


class MultiWaveletTransform(nn.Module):
    """A 1D MultiWavelet Transform module.

    This module applies a multi wavelet transform to its inputs. It's particularly
    useful for tasks involving time-series or signal processing in neural networks.

    Args:
        ich (int): The number of input channels.
        k (int): A parameter specifying the size of a transformation aspect (e.g., kernel size).
        alpha (int): A parameter used in the MultiWavelet Transform.
        c (int): Channel multiplier factor.
        nCZ (int): Number of MWT_CZ1d modules to use.
        L (int): A parameter related to the layer or length aspect of the transform.
        base (str): The type of base used in the transform, e.g., 'legendre'.
        attention_dropout (float): Dropout rate for the attention mechanism (not currently used in the module).
    """

    def __init__(
        self,
        ich: int = 1,
        k: int = 8,
        alpha: int = 16,
        c: int = 128,
        nCZ: int = 1,
        L: int = 0,
        base: str = "legendre",
        attention_dropout: float = 0.1,
    ):
        super(MultiWaveletTransform, self).__init__()
        print("base", base)
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.ich = ich
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.MWT_CZ = nn.ModuleList(MWT_CZ1d(k, alpha, L, c, base) for i in range(nCZ))

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> tuple[Any, None]:
        """Forward pass for the MultiWaveletTransform.

        Processes the queries, keys, and values through the multi wavelet transform.

        Args:
            queries (torch.Tensor): The query tensor.
            keys (torch.Tensor): The key tensor.
            values (torch.Tensor): The value tensor.
            attn_mask (torch.Tensor): The attention mask tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # Adjusting the shapes of keys and values to match queries
        if L > S:
            zeros = torch.zeros_like(queries[:, : (L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        values = values.view(B, L, -1)

        # Applying the multiwavelet transform
        V = self.Lk0(values).view(B, L, self.c, -1)
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
            if i < self.nCZ - 1:
                V = F.relu(V)

        # Final linear transformation and reshaping
        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, -1, D)
        return V.contiguous(), None


class MWT_CZ1d(nn.Module):
    """A 1D MultiWavelet Transform Component (MWT_CZ1d) module.

    This module is part of the MultiWavelet Transform architecture, responsible for
    applying the wavelet transform and its inverse.

    Args:
        k (int): Parameter specifying the size of transformation.
        alpha (int): Parameter used in the wavelet transform.
        L (int): A parameter related to the layer or length aspect of the transform.
        c (int): Channel multiplier factor.
        base (str): The type of base used in the transform, e.g., 'legendre'.
        initializer (callable): Optional initializer function for the module's weights.
        kwargs (dict): Additional keyword arguments.
    """

    def __init__(
        self,
        k: int = 3,
        alpha: int = 64,
        L: int = 0,
        c: int = 1,
        base: str = "legendre",
        initializer=None,
        **kwargs
    ):
        super(MWT_CZ1d, self).__init__()

        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        # Thresholding small values to zero for stability
        threshold = 1e-8
        H0r[np.abs(H0r) < threshold] = 0
        H1r[np.abs(H1r) < threshold] = 0
        G0r[np.abs(G0r) < threshold] = 0
        G1r[np.abs(G1r) < threshold] = 0

        # Sparse kernel transformations
        self.A = SparseKernelFT1d(k, alpha, c)
        self.B = SparseKernelFT1d(k, alpha, c)
        self.C = SparseKernelFT1d(k, alpha, c)

        # Linear transformation
        self.T0 = nn.Linear(k, k)

        # Registering buffers for efficient processing
        self.register_buffer("ec_s", torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer("ec_d", torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))
        self.register_buffer("rc_e", torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer("rc_o", torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MWT_CZ1d module.

        Processes the input tensor through the multi wavelet transform and its inverse.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        B, N, c, k = x.shape
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_x = x[:, 0 : nl - N, :, :]
        x = torch.cat([x, extra_x], 1)
        Ud = []
        Us = []

        # Decompose
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud.append(self.A(d) + self.B(x))
            Us.append(self.C(d))
        x = self.T0(x)  # coarsest scale transform

        # Reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        x = x[:, :N, :, :]

        return x

    def wavelet_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the wavelet transform on the input tensor.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of detail and smooth coefficients.
        """
        xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reorganize the tensor for even and odd components.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: Reorganized tensor.
        """
        B, N, c, ich = x.shape
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x_new = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x_new[..., ::2, :, :] = x_e
        x_new[..., 1::2, :, :] = x_o
        return x_new


class SparseKernelFT1d(nn.Module):
    """A 1D Sparse Kernel Fourier Transform module.

    This module is designed to apply a sparse kernel Fourier transform to its inputs.
    It's particularly useful for frequency domain processing in neural networks.

    Args:
        k (int): Kernel size or a related parameter.
        alpha (int): Specifies the number of Fourier modes to consider.
        c (int): Channel multiplier factor. Defaults to 1.
        nl (int): A layer-related parameter, currently not used in the module. Defaults to 1.
        initializer (callable): Optional initializer function for the module's weights.
        kwargs (dict): Additional keyword arguments.
    """

    def __init__(
        self, k: int, alpha: int, c: int = 1, nl: int = 1, initializer=None, **kwargs
    ):
        super(SparseKernelFT1d, self).__init__()

        self.modes1 = alpha
        self.scale = 1 / (c * k * c * k)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat)
        )
        self.weights1.requires_grad = True
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the SparseKernelFT1d module.

        Applies the sparse kernel Fourier transform to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        B, N, c, k = x.shape

        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        fourier_mode_limit = min(self.modes1, N // 2 + 1)
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :fourier_mode_limit] = complex_mul1d(
            x_fft[:, :, :fourier_mode_limit], self.weights1[:, :, :fourier_mode_limit]
        )

        # Inverse FFT and reshape
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


class MultiWaveletCross(nn.Module):
    """A 1D Multi wavelet Cross Attention layer.

    This layer applies multi wavelet transform followed by cross attention
    mechanisms for sequences of different lengths.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        seq_len_q (int): The length of the input sequence for the query.
        seq_len_kv (int): The length of the input sequence for key and value.
        modes (int): The number of frequency modes to use.
        c (int): Channel multiplier factor. Defaults to 64.
        k (int): Kernel size or related parameter. Defaults to 8.
        ich (int): Number of intermediate channels. Defaults to 512.
        L (int): A parameter related to the layer or length aspect of the transform.
        base (str): The type of base used in the transform, e.g., 'legendre'.
        mode_select_method (str): Method for selecting frequency modes ('random' or 'lowest').
        initializer (callable): Optional initializer function for the module's weights.
        activation (str): Activation function to use in the attention mechanism ('tanh' or other).
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len_q: int,
        seq_len_kv: int,
        modes: int,
        c: int = 64,
        k: int = 8,
        ich: int = 512,
        L: int = 0,
        base: str = "legendre",
        mode_select_method: str = "random",
        initializer=None,
        activation: str = "tanh",
        **kwargs
    ):
        super(MultiWaveletCross, self).__init__()
        print("base", base)

        self.c = c
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.attn1 = FourierCrossAttentionW(
            in_channels=in_channels,
            out_channels=out_channels,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            modes=modes,
            activation=activation,
            mode_select_method=mode_select_method,
        )
        self.attn2 = FourierCrossAttentionW(
            in_channels=in_channels,
            out_channels=out_channels,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            modes=modes,
            activation=activation,
            mode_select_method=mode_select_method,
        )
        self.attn3 = FourierCrossAttentionW(
            in_channels=in_channels,
            out_channels=out_channels,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            modes=modes,
            activation=activation,
            mode_select_method=mode_select_method,
        )
        self.attn4 = FourierCrossAttentionW(
            in_channels=in_channels,
            out_channels=out_channels,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            modes=modes,
            activation=activation,
            mode_select_method=mode_select_method,
        )
        self.T0 = nn.Linear(k, k)
        self.register_buffer("ec_s", torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer("ec_d", torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer("rc_e", torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer("rc_o", torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)
        self.modes1 = modes

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass of the MultiWaveletCross layer.

        Processes the input queries, keys, and values through wavelet transformations
        and attention mechanisms.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): An optional mask tensor for attention.

        Returns:
            Tuple[torch.Tensor, None]: The processed tensor and None.
        """
        B, N, H, E = q.shape  # (B, N, H, E) torch.Size([3, 768, 8, 2])
        _, S, _, _ = k.shape  # (B, S, H, E) torch.Size([3, 96, 8, 2])

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)

        if N > S:
            zeros = torch.zeros_like(q[:, : (N - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]

        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0 : nl - N, :, :]
        extra_k = k[:, 0 : nl - N, :, :]
        extra_v = v[:, 0 : nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)

        Ud_q = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_k = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_v = torch.jit.annotate(List[Tuple[Tensor]], [])

        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose
        for i in range(ns - self.L):
            # print('q shape',q.shape)
            d, q = self.wavelet_transform(q)
            Ud_q += [tuple([d, q])]
            Us_q += [d]
        for i in range(ns - self.L):
            d, k = self.wavelet_transform(k)
            Ud_k += [tuple([d, k])]
            Us_k += [d]
        for i in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            Ud_v += [tuple([d, v])]
            Us_v += [d]
        for i in range(ns - self.L):
            dk, sk = Ud_k[i], Us_k[i]
            dq, sq = Ud_q[i], Us_q[i]
            dv, sv = Ud_v[i], Us_v[i]
            Ud += [
                self.attn1(dq[0], dk[0], dv[0], mask)[0]
                + self.attn2(dq[1], dk[1], dv[1], mask)[0]
            ]
            Us += [self.attn3(sq, sk, sv, mask)[0]]
        v = self.attn4(q, k, v, mask)[0]

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        return v.contiguous(), None

    def wavelet_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies wavelet transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor to be transformed.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed detail and smooth coefficients.
        """
        xa = torch.cat(
            [
                x[:, ::2, :, :],
                x[:, 1::2, :, :],
            ],
            -1,
        )
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x: torch.Tensor) -> torch.Tensor:
        """Reorganizes the tensor into even and odd components for wavelet reconstruction.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: Reorganized tensor.
        """
        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


class FourierCrossAttentionW(nn.Module):
    """Fourier Cross Attention with wavelet (FourierCrossAttentionW) layer.

    This layer applies a Fourier transform-based cross-attention mechanism, which can be
    used in sequence-to-sequence models. It is designed to work with inputs transformed
    by a wavelet-based approach.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        seq_len_q (int): Sequence length of the query.
        seq_len_kv (int): Sequence length of the key and value.
        modes (int): The number of Fourier modes to use. Defaults to 16.
        activation (str): The activation function to be used. Defaults to 'tanh'.
        mode_select_method (str): Method for selecting frequency modes. Defaults to 'random'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len_q: int,
        seq_len_kv: int,
        modes: int = 16,
        activation: str = "tanh",
        mode_select_method: str = "random",
    ):
        super(FourierCrossAttentionW, self).__init__()
        print("Cross fourier correlation used!")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.activation = activation

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass of the FourierCrossAttentionW layer.

        Processes the input query, key, and value tensors through Fourier transform-based
        cross-attention mechanism.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): An optional mask tensor for attention.

        Returns:
            Tuple[torch.Tensor, None]: The processed tensor and None.
        """
        B, L, E, H = q.shape

        xq = q.permute(0, 3, 2, 1)  # size = [B, H, E, L] torch.Size([3, 8, 64, 512])
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(
            B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat
        )
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(
            B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat
        )
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)
        if self.activation == "tanh":
            xqk_ft = xqk_ft.tanh()
        elif self.activation == "softmax":
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception(
                "{} actiation function is not implemented".format(self.activation)
            )
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)

        xqkvw = xqkv_ft
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        out = torch.fft.irfft(
            out_ft / self.in_channels / self.out_channels, n=xq.size(-1)
        ).permute(0, 3, 2, 1)
        # size = [B, L, H, E]
        return out, None
