#!/usr/bin/env python
# coding: utf-8

"""Decoder for the Transformer."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqts.model.layers.components import (
    AddAndNorm,
    ResidualConnection,
    FeedForward,
    MultiHeadAttention,
)


class DecoderBlock(nn.Module):
    """Decoder Block module for the Transformer."""

    def __init__(
        self,
        dropout: float,
        d_model: int,
        activation: str = "relu",
        self_attention: nn.Module = MultiHeadAttention,
        cross_attention: nn.Module = MultiHeadAttention,
        feed_forward: nn.Module = FeedForward,
    ) -> None:
        """Initialize the Decoder Block module.

        Args:
            dropout (float): Dropout probability.
            d_model (int): Embedding dimension.
            activation (str, optional): Activation function. Defaults to "relu".
            self_attention (nn.Module, optional): Self attention module. Defaults to MultiHeadAttention.
            cross_attention (nn.Module, optional): Cross attention module. Defaults to MultiHeadAttention.
            feed_forward (nn.Module, optional): Feed forward module. Defaults to FeedForward.
        """
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout, d_model) for _ in range(3)]
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
            x, lambda x: self.self_attention(x, x, x, dec_mask)
        )
        x = self.residual_connection[1](
            x,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, enc_mask),
        )
        x = self.residual_connection[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """Decoder module for the Transformer."""

    def __init__(self, layers: nn.ModuleList, d_model: int) -> None:
        """Initialize the Decoder module.

        Args:
            layers (nn.ModuleList): List of DecoderBlock layers.
            d_model (int): Embedding dimension.
        """
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = AddAndNorm(d_model)

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
            x = layer(x, encoder_output, enc_mask, dec_mask)
        return self.norm(x)
