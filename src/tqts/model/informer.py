#!/usr/bin/env python
# coding: utf-8

"""Informer model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn

# import torch.nn.functional as F

from tqts.model.informers.layers.extensions import (
    DataEmbedding,
    ProbAttention,
    FullAttention,
    AttentionLayer,
    ConvLayer,
)

from tqts.model.informers.layers.encoder import Encoder, EncoderLayer, EncoderBlock
from tqts.model.informers.layers.decoder import Decoder, DecoderLayer


class Informer(nn.Module):
    """Informer Model."""

    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        out_len: int,
        factor: int = 5,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        attn: str = "prob",
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        distil: bool = False,
        mix: bool = False,
        device: str = "cpu",
    ) -> None:
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Attention
        attention = ProbAttention if attn == "prob" else FullAttention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        attention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        attention(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=mix,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

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
        """Forward pass of the Informer model.

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
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )

        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]


class InformerStack(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=None,
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        attn="prob",
        embed="fixed",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
    ):
        super(InformerStack, self).__init__()
        if e_layers is None:
            e_layers = [3, 2, 1]
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Attention
        Attn = ProbAttention if attn == "prob" else FullAttention

        # Encoder
        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(
                                False,
                                factor,
                                attention_dropout=dropout,
                                output_attention=output_attention,
                            ),
                            d_model,
                            n_heads,
                            mix=False,
                        ),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for _ in range(el)
                ],
                [ConvLayer(d_model) for _ in range(el - 1)] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model),
            )
            for el in e_layers
        ]
        self.encoder = EncoderBlock(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=mix,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

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
        """Forward pass of the Informer model.

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
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]
