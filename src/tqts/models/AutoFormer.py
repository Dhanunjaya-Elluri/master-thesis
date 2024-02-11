#!/usr/bin/env python
# coding: utf-8

"""AutoFormer models."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"


import torch
import torch.nn as nn

from tqts.models.layers.autocorrelation import AutoCorrelation, AutoCorrelationLayer
from tqts.models.layers.auxiliary import MyLayerNorm, SeriesDeComp
from tqts.models.layers.decoder import AutoDecoder, AutoDecoderLayer
from tqts.models.layers.embedding import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_pos_temp,
    DataEmbedding_wo_temp,
)
from tqts.models.layers.encoder import AutoEncoder, AutoEncoderLayer


class Model(nn.Module):
    """
    AutoFormer is a neural network models for time series forecasting.
    It uses a series-wise connection with inherent O(LlogL) complexity for efficient processing.

    Args:
        configs: Configuration object containing models settings.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # series decomposition
        kernel_size = configs.moving_avg
        self.decomp = SeriesDeComp(kernel_size=kernel_size)

        # embedding layers_temp
        self.enc_embedding, self.dec_embedding = self._select_embedding(configs)

        # Encoder
        self.encoder = AutoEncoder(
            [
                AutoEncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
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
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=MyLayerNorm(configs.d_model),
        )

        # Decoder
        self.decoder = AutoDecoder(
            [
                AutoDecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=MyLayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    @staticmethod
    def _select_embedding(configs):
        """
        Selects and initializes the appropriate embedding layer based on the configuration.

        Args:
            configs: Configuration object containing models settings.

        Returns:
            Tuple of (encoder_embedding, decoder_embedding)
        """
        embedding_types = {
            0: (DataEmbedding_wo_pos, DataEmbedding_wo_pos),
            1: (DataEmbedding, DataEmbedding),
            2: (DataEmbedding_wo_pos, DataEmbedding_wo_pos),
            3: (DataEmbedding_wo_temp, DataEmbedding_wo_temp),
            4: (DataEmbedding_wo_pos_temp, DataEmbedding_wo_pos_temp),
        }
        enc_embedding_class, dec_embedding_class = embedding_types.get(
            configs.embed_type, (DataEmbedding, DataEmbedding)
        )
        enc_embedding = enc_embedding_class(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        dec_embedding = dec_embedding_class(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        return enc_embedding, dec_embedding

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        """
        Forward pass of the Autoformer models.

        Args:
            x_enc: Input sequence for the encoder.
            x_mark_enc: Temporal encoding for the encoder input.
            x_dec: Input sequence for the decoder.
            x_mark_dec: Temporal encoding for the decoder input.
            enc_self_mask: Self-attention mask for the encoder.
            dec_self_mask: Self-attention mask for the decoder.
            dec_enc_mask: Cross-attention mask between decoder and encoder.

        Returns:
            The output of the models and, if requested, attention weights.
        """
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device
        )
        seasonal_init, trend_init = self.decomp(x_enc)

        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len :, :], zeros], dim=1
        )

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init,
        )

        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
