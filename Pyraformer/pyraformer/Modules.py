#!/usr/bin/env python
# coding: utf-8

"""Layers section for Pyraformer model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module used in Transformer models.

    This attention mechanism scales the dot products by the square root of the dimensionality of the
    key vectors, which helps in stabilizing the gradients. It's commonly used in the multi-head
    attention mechanism of Transformer models.

    Args:
        temperature (float): The scaling factor for the dot product, usually the square root of
                             the key dimension.
        attn_dropout (float): Dropout rate for the attention scores. Default: 0.2.
    """

    def __init__(self, temperature: float, attn_dropout: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None
    ) -> (Tensor, Tensor):
        """
        Forward pass of the ScaledDotProductAttention.

        Args:
            q (Tensor): Query tensor of shape (batch_size, n_heads, seq_len, d_k).
            k (Tensor): Key tensor of shape (batch_size, n_heads, seq_len, d_k).
            v (Tensor): Value tensor of shape (batch_size, n_heads, seq_len, d_v).
            mask (Tensor, optional): Mask tensor of shape (batch_size, 1, 1, seq_len).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Output tensor of shape (batch_size, n_heads, seq_len, d_v).
                - Attention tensor of shape (batch_size, n_heads, seq_len, seq_len).
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
