#!/usr/bin/env python
# coding: utf-8

"""Transformer Main Module"""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn

from tqts.model.layers.decoder import Decoder
from tqts.model.layers.encoder import Encoder
from tqts.model.layers.components import (
    InputEmbeddings,
    PositionalEncoding,
    LinearLayer,
)


class Transformer(nn.Module):
    """Transformer module."""

    def __init__(
        self,
        encoder: nn.Module = Encoder,
        decoder: nn.Module = Decoder,
        enc_emb: nn.Module = InputEmbeddings,
        dec_emb: nn.Module = InputEmbeddings,
        enc_pos: nn.Module = PositionalEncoding,
        dec_pos: nn.Module = PositionalEncoding,
        linear_layer: nn.Module = LinearLayer,
    ):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_emb = enc_emb
        self.dec_emb = dec_emb
        self.enc_pos = enc_pos
        self.dec_pos = dec_pos
        self.linear_layer = linear_layer

    def encode(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size).
            mask (torch.Tensor, optional): Mask tensor of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        x = self.enc_emb(x)
        x = self.enc_pos(x)
        return self.encoder(x, mask)

    def decode(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        enc_mask: torch.Tensor = None,
        dec_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of the Decoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size).
            encoder_output (torch.Tensor): Encoder output tensor of shape (seq_len, batch_size, d_model).
            enc_mask (torch.Tensor, optional): Encoder mask tensor of shape (seq_len, seq_len). Defaults to None.
            dec_mask (torch.Tensor, optional): Decoder mask tensor of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        x = self.dec_emb(x)
        x = self.dec_pos(x)
        return self.decoder(x, encoder_output, enc_mask, dec_mask)

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Linear layer module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, output_size).
        """
        return self.linear_layer(x)
