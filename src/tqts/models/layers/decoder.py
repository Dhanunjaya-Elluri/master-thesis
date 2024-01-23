#!/usr/bin/env python
# coding: utf-8

"""Decoder module for TQTS Model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

from typing import Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tqts.models.layers.auxiliary import SeriesDeComp


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

    def __init__(
        self, layers: list, norm_layer: nn.Module = None, projection=None, **_
    ) -> None:
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: torch.Tensor = None,
        cross_mask: torch.Tensor = None,
        **kwargs
    ) -> tuple[Tensor | Any, list[Any]]:
        """Forward pass of the Decoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            cross (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            x_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            cross_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            tuple: Output tensor of shape (batch_size, seq_len, d_model) and attention tensor of shape (batch_size, seq_len, seq_len).
        """
        attentions = []
        for layer in self.layers:
            x, a_sa, a_ca = layer(x, cross, x_mask, cross_mask, **kwargs)
            attentions.append(a_sa)
            attentions.append(a_ca)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, attentions


class AutoDecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        c_out,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu",
    ):
        super(AutoDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False
        )
        self.decomp1 = SeriesDeComp(moving_avg)
        self.decomp2 = SeriesDeComp(moving_avg)
        self.decomp3 = SeriesDeComp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        )
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(
            1, 2
        )
        return x, residual_trend


class AutoDecoder(nn.Module):
    """
    Autoformer Decoder
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(AutoDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
