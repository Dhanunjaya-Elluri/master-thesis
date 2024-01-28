#!/usr/bin/env python
# coding: utf-8

"""Encoder module for TQTS Model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqts.models.layers.auxiliary import SeriesDeComp


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Encoder Block module for the Transformer Model.

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


class AutoEncoderLayer(nn.Module):
    """
    AutoFormer encoder layer with the progressive decomposition architecture
    """

    def __init__(
        self,
        attention,
        d_model,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu",
    ):
        super(AutoEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False
        )
        self.decomp1 = SeriesDeComp(moving_avg)
        self.decomp2 = SeriesDeComp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class AutoEncoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(AutoEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class LogSparseEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        LogSparseEncoderLayer is a variant of the encoder layer in the paper
        "LogSparse Transformer: Towards Efficient Transformers" (https://arxiv.org/abs/2012.11747)

        Args:
            attention: attention module
            d_model: dimension of the model
            d_ff: dimension of the feedforward layer
            dropout: dropout rate
            activation: activation function
        """
        super(LogSparseEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, **kwargs):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, **kwargs)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class LogSparseEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        """
        LogSparseEncoder is a variant of the encoder in the paper
        "LogSparse Transformer: Towards Efficient Transformers" (https://arxiv.org/abs/2012.11747)

        Args:
            attn_layers: list of attention layers
            conv_layers: list of convolution layers
            norm_layer: normalization layer
        """
        super(LogSparseEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, **kwargs):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask, **kwargs)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, **kwargs)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
