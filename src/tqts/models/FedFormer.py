#!/usr/bin/env python
# coding: utf-8

"""FedFormer Transformer Module"""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqts.models.layers.autocorrelation import AutoCorrelationLayer
from tqts.models.layers.auxiliary import MyLayerNorm, SeriesDeComp, SeriesDeCompMulti
from tqts.models.layers.decoder import AutoDecoder, AutoDecoderLayer
from tqts.models.layers.embedding import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_pos_temp,
    DataEmbedding_wo_temp,
)
from tqts.models.layers.encoder import AutoEncoder, AutoEncoderLayer
from tqts.models.layers.fourier_correlation import FourierBlock, FourierCrossAttention
from tqts.models.layers.multi_wavelet_corr import (
    MultiWaveletCross,
    MultiWaveletTransform,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """FedFormer models that performs attention mechanism in the frequency domain, achieving O(N) complexity.
    It decomposes the input series into trend and seasonal components and applies distinct attention mechanisms
    to these components.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.de_comp = SeriesDeCompMulti(kernel_size)
        else:
            self.de_comp = SeriesDeComp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        # self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                           configs.dropout)
        # self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                           configs.dropout)
        if configs.embed_type == 0:
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

        if configs.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base
            )
            decoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base
            )
            decoder_cross_att = MultiWaveletCross(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=configs.modes,
                ich=configs.d_model,
                base=configs.base,
                activation=configs.cross_activation,
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select,
            )
            decoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len // 2 + self.pred_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select,
            )
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len // 2))
        dec_modes = int(
            min(configs.modes, (configs.seq_len // 2 + configs.pred_len) // 2)
        )
        print("enc_modes: {}, dec_modes: {}".format(enc_modes, dec_modes))

        self.encoder = AutoEncoder(
            [
                AutoEncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, configs.d_model, configs.n_heads
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
                        decoder_self_att, configs.d_model, configs.n_heads
                    ),
                    AutoCorrelationLayer(
                        decoder_cross_att, configs.d_model, configs.n_heads
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
        """Forward pass of the FEDFormer models.

        Processes the input series through decomposition, embedding, encoder, and decoder layers to produce a forecast.

        Args:
            x_enc (torch.Tensor): Input tensor for the encoder.
            x_mark_enc (torch.Tensor): Additional markers or features for the encoder input.
            x_dec (torch.Tensor): Input tensor for the decoder.
            x_mark_dec (torch.Tensor): Additional markers or features for the decoder input.
            enc_self_mask (torch.Tensor, optional): Self-attention mask for the encoder.
            dec_self_mask (torch.Tensor, optional): Self-attention mask for the decoder.
            dec_enc_mask (torch.Tensor, optional): Cross-attention mask between decoder and encoder.

        Returns:
            tuple[Any, Any] | Any: Output tensor of shape (batch_size, seq_len, d_model) and attention tensor of shape (batch_size, seq_len, seq_len).
        """
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(
        #     device
        # )  # cuda()
        seasonal_init, trend_init = self.de_comp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len :, :], (0, 0, 0, self.pred_len)
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
