#!/usr/bin/env python
# coding: utf-8

"""Encoder layer for vanilla Transformer."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqts.model.layers.extensions import (
    AddAndNorm,
    MultiHeadAttention,
    FeedForward,
    ResidualConnection,
)


class EncoderBlock(nn.Module):
    """Encoder Block module for the Transformer."""

    def __init__(
        self,
        dropout: float,
        d_model: int,
        activation: str = "relu",
        attention: nn.Module = MultiHeadAttention,
        feed_forward: nn.Module = FeedForward,
    ) -> None:
        """Initialize the Encoder Block module.

        Args:
            dropout (float): Dropout probability.
            d_model (int): Embedding dimension.
            activation (str, optional): Activation function. Defaults to "relu".
            attention (nn.Module, optional): Attention module. Defaults to MultiHeadAttention.
            feed_forward (nn.Module, optional): Feed forward module. Defaults to FeedForward.
        """
        super(EncoderBlock, self).__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout, d_model) for _ in range(2)]
        )
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor, enc_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the Encoder Block module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            enc_mask (torch.Tensor, optional): Mask tensor of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        x = self.residual_connection[0](x, lambda x: self.attention(x, x, x, enc_mask))
        x = self.residual_connection[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """Encoder module for the Transformer."""

    def __init__(self, layers: nn.ModuleList, d_model: int) -> None:
        """Initialize the Encoder module.

        Args:
            layers (nn.ModuleList): List of EncoderBlock layers.
            d_model (int): Embedding dimension.
        """
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = AddAndNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
