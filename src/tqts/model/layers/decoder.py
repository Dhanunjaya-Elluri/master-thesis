#!/usr/bin/env python
# coding: utf-8

"""Decoder for the Transformer."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqts.model.layers.components import (
    AddNorm,
    ResidualConnection,
    FeedForward,
    MultiHeadAttention,
)


class DecoderBlock(nn.Module):
    """Decoder Block module for the Transformer."""

    def __init__(
        self,
        dropout: float,
        activation: str = "relu",
        attention: nn.Module = MultiHeadAttention,
        feed_forward: nn.Module = FeedForward,
    ):
        super(DecoderBlock, self).__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )
        self.activation = getattr(F, activation)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        enc_mask: torch.Tensor = None,
        dec_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of the Decoder Block module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            encoder_output (torch.Tensor): Encoder output tensor of shape (seq_len, batch_size, d_model).
            enc_mask (torch.Tensor, optional): Encoder mask tensor of shape (seq_len, seq_len). Defaults to None.
            dec_mask (torch.Tensor, optional): Decoder mask tensor of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        x = self.residual_connection[0](
            x, lambda x: self.attention(x, x, x, mask=dec_mask)
        )
        x = self.residual_connection[1](
            x,
            lambda x: self.attention(x, encoder_output, encoder_output, mask=enc_mask),
        )
        x = self.residual_connection[2](x, self.feed_forward)
        return self.activation(x)


class Decoder(nn.Module):
    """Decoder module for the Transformer."""

    def __init__(self, layers: nn.ModuleList):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = AddNorm()

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        enc_mask: torch.Tensor = None,
        dec_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of the Decoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            encoder_output (torch.Tensor): Encoder output tensor of shape (seq_len, batch_size, d_model).
            enc_mask (torch.Tensor, optional): Encoder mask tensor of shape (seq_len, seq_len). Defaults to None.
            dec_mask (torch.Tensor, optional): Decoder mask tensor of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        for layer in self.layers:
            x = layer(x, encoder_output, enc_mask=enc_mask, dec_mask=dec_mask)
        return self.norm(x)
