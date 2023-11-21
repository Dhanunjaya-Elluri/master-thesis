#!/usr/bin/env python
# coding: utf-8

"""Transformer Main Module"""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn

from tqts.model.layers.decoder import Decoder, DecoderBlock
from tqts.model.layers.encoder import Encoder, EncoderBlock
from tqts.model.layers.components import (
    InputEmbeddings,
    PositionalEncoding,
    LinearLayer,
    MultiHeadAttention,
    FeedForward,
)


class Transformer(nn.Module):
    """Transformer module."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super(Transformer, self).__init__()

        # Embeddings and positional encoding
        self.input_embeddings = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Encoder
        encoder_layers = nn.ModuleList(
            [
                EncoderBlock(
                    dropout,
                    d_model,
                    activation,
                    MultiHeadAttention(d_model, num_heads, dropout),
                    FeedForward(d_model, dim_feedforward, dropout),
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.encoder = Encoder(encoder_layers, d_model)

        # Decoder
        decoder_layers = nn.ModuleList(
            [
                DecoderBlock(
                    dropout,
                    d_model,
                    activation,
                    MultiHeadAttention(d_model, num_heads, dropout),
                    MultiHeadAttention(d_model, num_heads, dropout),
                    FeedForward(d_model, dim_feedforward, dropout),
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.decoder = Decoder(decoder_layers, d_model)

        # Output
        self.output = LinearLayer(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of the Transformer module.

        Args:
            src (torch.Tensor): Source input tensor of shape (seq_len, batch_size).
            tgt (torch.Tensor): Target input tensor of shape (seq_len, batch_size).
            src_mask (torch.Tensor, optional): Source mask tensor of shape (seq_len, seq_len). Defaults to None.
            tgt_mask (torch.Tensor, optional): Target mask tensor of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, vocab_size).
        """
        src = self.input_embeddings(src)
        src = self.positional_encoding(src)
        encoded = self.encoder(src, src_mask)
        tgt = self.input_embeddings(tgt)
        tgt = self.positional_encoding(tgt)
        output = self.decoder(tgt, encoded, src_mask, tgt_mask)
        return self.output(output)
