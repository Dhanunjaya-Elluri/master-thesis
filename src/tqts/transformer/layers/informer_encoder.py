#!/usr/bin/env python
# coding: utf-8

"""Informer: Encoder module for Transformer with ProbSpace Self-Attention Mechanism."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """
    Convolutional Layer for the Encoder.
    """

    def __init__(self, c_in, c_out, kernel_size, padding):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
        )
        self.activation = nn.ELU()
        self.norm = nn.BatchNorm1d(c_out)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Convolutional Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, c_out).
        """
        x = self.conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """Encoder Layer for the Transformer."""

    def __init__(
        self, d_model, n_heads, d_ff, dropout, attention, activation: str = "relu"
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_size=1,
            # padding=0,
            # padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff,
            out_channels=d_model,
            kernel_size=1,
            # padding=0,
            # padding_mode="circular",
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attention_mask=None):
        """
        Forward pass of the Encoder Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        if self.self_attention:
            x = x + self.dropout(self.self_attention(x, x, x, attention_mask))
            x = self.norm1(x)
        else:
            x = self.norm1(x)
            x = self.activation(self.conv1(x.transpose(1, 2)))
            x = self.conv2(x).transpose(1, 2)
            x = self.dropout(x)
        return x
