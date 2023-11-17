#!/usr/bin/env python
# coding: utf-8

"""Encoder layer for vanilla Transformer."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqts.model.layers.components import (
    AddNorm,
    MultiHeadAttention,
    FeedForward,
    ResidualConnection,
)


class EncoderBlock(nn.Module):
    """Encoder Block module for the Transformer."""

    def __init__(
        self,
        dropout: float,
        activation: str = "relu",
        attention: nn.Module = MultiHeadAttention,
        feed_forward: nn.Module = FeedForward,
    ):
        super(EncoderBlock, self).__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the Encoder Block module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        x = self.residual_connection[0](x, lambda x: self.attention(x, x, x, mask=mask))
        x = self.residual_connection[1](x, self.feed_forward)
        return self.activation(x)


class Encoder(nn.Module):
    """Encoder module for the Transformer."""

    def __init__(self, layers: nn.ModuleList):
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = AddNorm()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)
