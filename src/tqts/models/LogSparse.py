#!/usr/bin/env python
# coding: utf-8

"""LogSparse Transformer Module"""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

from typing import Any

import torch
import torch.nn as nn

from tqts.models.layers.attention import (
    FullAttention,
    LogSparseAttentionLayer,
)
from tqts.models.layers.decoder import Decoder, DecoderLayer
from tqts.models.layers.embedding import LogSparseDataEmbedding
from tqts.models.layers.encoder import Encoder, EncoderLayer


class Model(nn.Module):
    """LogSparse Transformer with O(L log(L)) complexity as described in the paper:
    'LogSparse Transformer' (https://arxiv.org/abs/1907.00235).

    This models aims to reduce the computational complexity of standard Transformers
    by applying logarithmic sparsity in the attention mechanism.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.total_length = configs.pred_len + configs.seq_len

        # Embedding
        self.enc_embedding = LogSparseDataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
            kernel_size=configs.kernel_size,
        )
        self.dec_embedding = LogSparseDataEmbedding(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
            kernel_size=configs.kernel_size,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    LogSparseAttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                            sparse_flag=configs.sparse_flag,
                            win_len=configs.win_len,
                            res_len=configs.res_len,
                        ),
                        configs.d_model,
                        configs.n_heads,
                        configs.qk_ker,
                        v_conv=configs.v_conv,
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
                    LogSparseAttentionLayer(
                        FullAttention(
                            True,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                            sparse_flag=configs.sparse_flag,
                            win_len=configs.win_len,
                            res_len=configs.res_len,
                        ),
                        configs.d_model,
                        configs.n_heads,
                        configs.qk_ker,
                        v_conv=configs.v_conv,
                    ),
                    LogSparseAttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                            sparse_flag=configs.sparse_flag,
                            win_len=configs.win_len,
                            res_len=configs.res_len,
                        ),
                        configs.d_model,
                        configs.n_heads,
                        configs.qk_ker,
                        v_conv=configs.v_conv,
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
    ) -> tuple[Any, list[Any]] | Any:
        """
        Forward pass of the LogSparse Transformer models.

        Processes the input through the encoder and decoder to produce the forecast.

        Args:
            x_enc (torch.Tensor): Input tensor for the encoder.
            x_mark_enc (torch.Tensor): Additional markers or features for the encoder input.
            x_dec (torch.Tensor): Input tensor for the decoder.
            x_mark_dec (torch.Tensor): Additional markers or features for the decoder input.
            enc_self_mask (torch.Tensor, optional): Self-attention mask for the encoder.
            dec_self_mask (torch.Tensor, optional): Self-attention mask for the decoder.
            dec_enc_mask (torch.Tensor, optional): Cross-attention mask between decoder and encoder.

        Returns:
            torch.Tensor: The output tensor of the models. If output_attention is True, attention weights are also returned.
        """
        attentions = []
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, a = self.encoder(enc_out, attn_mask=enc_self_mask)
        attentions.append(a)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out, a = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )
        attentions.append(a)

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attentions
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
