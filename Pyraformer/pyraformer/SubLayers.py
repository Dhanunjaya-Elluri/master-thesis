#!/usr/bin/env python
# coding: utf-8

"""Sub Layers section for Pyraformer model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Pyraformer.pyraformer.Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    A Multi-Head Attention module typically used in Transformer models.

    This module performs attention computation in parallel across multiple heads,
    allowing the model to jointly attend to information from different representation
    subspaces at different positions.

    Args:
        n_head (int): Number of attention heads.
        d_model (int): Dimension of the model (input and output size).
        d_k (int): Dimension of keys/queries.
        d_v (int): Dimension of values.
        dropout (float, optional): Dropout rate. Default: 0.1.
        normalize_before (bool, optional): Whether to apply layer normalization before attention. Default: True.
    """

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        normalize_before: bool = True,
    ):
        super(MultiHeadAttention, self).__init__()
        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5, attn_dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Multi-Head Attention layer.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, len_q, d_model).
            k (torch.Tensor): Key tensor of shape (batch_size, len_k, d_model).
            v (torch.Tensor): Value tensor of shape (batch_size, len_v, d_model).
            mask (Optional[torch.Tensor]): Optional mask tensor for attention. Shape: (batch_size, len_q, len_k).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor after attention and the attention weights.
            Output tensor shape: (batch_size, len_q, d_model).
            Attention weights shape: (batch_size, n_head, len_q, len_k).
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """
    A two-layer position-wise feed-forward neural network.

    This component is another key part of the Transformer architecture, used immediately
    after the multi-head self-attention layer. It applies two linear transformations
    and a GELU activation in between.

    Args:
        d_in (int): Input dimension of the feed-forward network.
        d_hid (int): Hidden layer dimension.
        dropout (float, optional): Dropout rate. Default: 0.1.
        normalize_before (bool, optional): Whether to apply layer normalization before the feed-forward network. Default: True.
    """

    def __init__(
        self, d_in: int, d_hid: int, dropout: float = 0.1, normalize_before: bool = True
    ):
        super(PositionwiseFeedForward, self).__init__()
        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.layer_norm = GraphNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Positionwise Feed-Forward layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_in).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x
