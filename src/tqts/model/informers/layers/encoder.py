#!/usr/bin/env python
# coding: utf-8

"""Encoder module for Informer Model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    """Encoder Block module for the Informer Model."""

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Initialize the Encoder Block module.

        Args:
            d_model (int): Embedding dimension.
            d_ff (int): Feed forward dimension.
            attention (nn.Module): Attention module.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            activation (str, optional): Activation function. Defaults to "relu".
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple:
        """Forward pass of the Encoder Block module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            tuple: Output tensor of shape (batch_size, seq_len, d_model) and attention tensor of shape (batch_size, seq_len, seq_len).
        """
        new_x, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """Encoder module for the Informer Model."""

    def __init__(
        self, attn_layers: list, conv_layers: list = None, norm_layer: nn.Module = None
    ) -> None:
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm_layer = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple:
        """Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            tuple: Output tensor of shape (batch_size, seq_len, d_model) and attention tensor of shape (batch_size, seq_len, seq_len).
        """
        attentions = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask)
                x = conv_layer(x)
                attentions.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attentions.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attentions.append(attn)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x, attentions


class EncoderBlock(nn.Module):
    """Encoder Block module for the Informer Model."""

    def __init__(self, encoders: list, input_lens: list):
        super(EncoderBlock, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.input_lens = input_lens

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple:
        """Forward pass of the Encoder Block module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            tuple: Output tensor of shape (batch_size, seq_len, d_model) and attention tensor of shape (batch_size, seq_len, seq_len).
        """
        x_stacks, attentions = [], []
        for i_len, encoder in zip(self.input_lens, self.encoders):
            input_len = x.shape[1] // (2**i_len)
            x_s, attn = encoder(x[:, -input_len:, :])
            x_stacks.append(x_s)
            attentions.append(attn)
        x_stacks = torch.cat(x_stacks, -2)
        return x_stacks, attentions
