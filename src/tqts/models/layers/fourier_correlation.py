#!/usr/bin/env python
# coding: utf-8

"""Fourier Correlation Layers for FEDFormer models."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(
    seq_len: int, modes: int = 64, mode_select_method: str = "random"
) -> List[int]:
    """Get a specified number of frequency modes from a sequence.

    The method of selecting the modes can be either 'random' or 'lowest'.
    'random' selects the modes randomly, while 'lowest' selects the lowest frequency modes.

    Args:
        seq_len (int): The length of the sequence.
        modes (int): The number of frequency modes to select. Defaults to 64.
        mode_select_method (str): The method of selecting the modes.
                                  Options are 'random' or any other string for 'lowest'. Defaults to 'random'.

    Returns:
        List[int]: A list of selected frequency mode indices.
    """
    modes = min(modes, seq_len // 2)

    if mode_select_method == "random":
        index = list(range(seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(modes))

    index.sort()
    return index


def complex_mul1d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Perform complex multiplication in 1D.

    Args:
        x (torch.Tensor): The input tensor.
        weights (torch.Tensor): The weight tensor.

    Returns:
        torch.Tensor: The result of the complex multiplication.
    """
    return torch.einsum("bhi,hio->bho", x, weights)


class FourierBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        modes: int = 0,
        mode_select_method: str = "random",
    ):
        """A 1D Fourier block for representation learning on the frequency domain.

        It performs operations such as FFT (Fast Fourier Transform), a linear transform, and an Inverse FFT.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            seq_len (int): The length of the input sequences.
            modes (int): The number of frequency modes to use. Defaults to 0.
            mode_select_method (str): Method for selecting frequency modes ('random' or 'lowest'). Defaults to 'random'.
        """
        super(FourierBlock, self).__init__()
        print("Fourier enhanced block used!")

        # Get modes on frequency domain
        self.index = get_frequency_modes(
            seq_len, modes=modes, mode_select_method=mode_select_method
        )
        print(f"modes={modes}, index={self.index}")

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                8,
                in_channels // 8,
                out_channels // 8,
                len(self.index),
                dtype=torch.cfloat,
            )
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass of the FourierBlock.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            Tuple[torch.Tensor, None]: The transformed tensor and None.
        """
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)

        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            if i >= x_ft.shape[3] or wi >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, wi] = complex_mul1d(
                x_ft[:, :, :, i], self.weights1[:, :, :, wi]
            )

        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x, None


class FourierCrossAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len_q: int,
        seq_len_kv: int,
        modes: int = 64,
        mode_select_method: str = "random",
        activation: str = "tanh",
        policy: int = 0,
    ):
        """A 1D Fourier Cross Attention layer. This layer performs operations such as FFT (Fast Fourier Transform),
        linear transformation, an attention mechanism, and an Inverse FFT.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            seq_len_q (int): The length of the input sequences for queries.
            seq_len_kv (int): The length of the input sequences for keys and values.
            modes (int): The number of frequency modes to use. Defaults to 64.
            mode_select_method (str): Method for selecting frequency modes ('random' or 'lowest'). Defaults to 'random'.
            activation (str): The activation function to use ('tanh' or 'softmax'). Defaults to 'tanh'.
            policy (int): A parameter for potential future use; currently not used in the layer.
        """
        super(FourierCrossAttention, self).__init__()
        print("Fourier enhanced cross attention used!")

        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Get modes for queries and keys/values on frequency domain
        self.index_q = get_frequency_modes(
            seq_len_q, modes=modes, mode_select_method=mode_select_method
        )
        self.index_kv = get_frequency_modes(
            seq_len_kv, modes=modes, mode_select_method=mode_select_method
        )

        print(f"modes_q={len(self.index_q)}, index_q={self.index_q}")
        print(f"modes_kv={len(self.index_kv)}, index_kv={self.index_kv}")

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                8,
                in_channels // 8,
                out_channels // 8,
                len(self.index_q),
                dtype=torch.cfloat,
            )
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass of the FourierCrossAttention layer.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            Tuple[torch.Tensor, None]: The transformed tensor and None.
        """
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        # xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(
            B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat
        )
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            if j >= xq_ft.shape[3]:
                continue
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(
            B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat
        )
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            if j >= xk_ft.shape[3]:
                continue
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
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
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            if i >= xqkvw.shape[3] or j >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(
            out_ft / self.in_channels / self.out_channels, n=xq.size(-1)
        )
        return out, None
