#!/usr/bin/env python
# coding: utf-8

"""Decoder module for Informer Model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    """Decoder Block module for the Informer Model."""

    def __init__(
        self,
        self_attention: nn.Module,
        cross_attention: nn.Module,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super(DecoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: torch.Tensor = None,
        cross_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of the Decoder Block module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            cross (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            x_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            cross_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = x + self.dropout(self.self_attention(x, x, x, x_mask)[0])
        x = self.norm1(x)
        y = x = self.norm2(
            x + self.dropout(self.cross_attention(x, cross, cross, cross_mask)[0])
        )
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    """Decoder module for the Informer Model."""

    def __init__(self, layers: list, norm_layer: nn.Module = None) -> None:
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: torch.Tensor = None,
        cross_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of the Decoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            cross (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            x_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            cross_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, cross, x_mask, cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x
