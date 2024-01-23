#!/usr/bin/env python
# coding: utf-8

"""Transformer Main Module"""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

from typing import Any

import torch
import torch.nn as nn

from tqts.models.layers.attention import AttentionLayer, FullAttention
from tqts.models.layers.decoder import Decoder, DecoderLayer
from tqts.models.layers.embedding import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_temp,
    DataEmbedding_wo_pos_temp,
)
from tqts.models.layers.encoder import Encoder, EncoderLayer


class Model(nn.Module):
    """
    A vanilla Transformer models with O(L^2) complexity designed for time series forecasting.

    This models includes both an encoder and a decoder, along with various embedding options
    based on the configuration. It is capable of handling different types of embeddings such as
    positional, temporal, or their combinations.

    Attributes:
    pred_len (int): Length of the prediction.
    output_attention (bool): Flag to determine if the attention weights are output.
    enc_embedding (nn.Module): Embedding layer for the encoder.
    dec_embedding (nn.Module): Embedding layer for the decoder.
    encoder (Encoder): Transformer encoder module.
    decoder (Decoder): Transformer decoder module.

    Parameters:
    configs (object): Configuration object containing models settings.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.dec_embedding = DataEmbedding(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.dec_embedding = DataEmbedding(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.dec_embedding = DataEmbedding_wo_pos(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.dec_embedding = DataEmbedding_wo_temp(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.dec_embedding = DataEmbedding_wo_pos_temp(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: torch.Tensor = None,
        dec_self_mask: torch.Tensor = None,
        dec_enc_mask: torch.Tensor = None,
    ) -> tuple[Any, Any] | Any:
        """Forward pass of the Transformer models.

        Args:
            x_enc (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            x_mark_enc (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            x_dec (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            x_mark_dec (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            enc_self_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            dec_self_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            dec_enc_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
