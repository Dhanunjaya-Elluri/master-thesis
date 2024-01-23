#!/usr/bin/env python
# coding: utf-8

"""Informer models."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn

from tqts.models.layers.attention import ProbAttention, AttentionLayer
from tqts.models.layers.auxiliary import ConvLayer
from tqts.models.layers.decoder import Decoder, DecoderLayer
from tqts.models.layers.embedding import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_temp,
    DataEmbedding_wo_pos_temp,
)
from tqts.models.layers.encoder import Encoder, EncoderLayer


class Model(nn.Module):
    """Informer Model."""

    def __init__(
        self,
        configs,
    ) -> None:
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
                        ProbAttention(
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
            [ConvLayer(configs.d_model) for l in range(configs.e_layers - 1)]
            if configs.distil
            else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        ProbAttention(
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
    ) -> tuple:
        """Forward pass of the Informer models.

        Args:
            x_enc (torch.Tensor): Input tensor of shape (batch_size, seq_len, enc_in).
            x_mark_enc (torch.Tensor): Input tensor of shape (batch_size, seq_len, 4).
            x_dec (torch.Tensor): Input tensor of shape (batch_size, seq_len, dec_in).
            x_mark_dec (torch.Tensor): Input tensor of shape (batch_size, seq_len, 4).
            enc_self_mask (torch.Tensor): Attention mask tensor. Defaults to None.
            dec_self_mask (torch.Tensor): Attention mask tensor. Defaults to None.
            dec_enc_mask (torch.Tensor): Attention mask tensor. Defaults to None.

        Returns:
            tuple: Output tensor of shape (batch_size, seq_len, c_out) and attention tensor of shape (batch_size, seq_len, seq_len).
        """
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attentions = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attentions
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
