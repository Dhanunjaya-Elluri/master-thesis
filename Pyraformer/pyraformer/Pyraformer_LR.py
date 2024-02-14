#!/usr/bin/env python
# coding: utf-8

"""Layers section for Pyraformer model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"


import torch
import torch.nn as nn
from pyraformer.embed import DataEmbedding
from pyraformer.Layers import (
    AvgPoolingConstruct,
    BottleneckConstruct,
    ConvConstruct,
    Decoder,
    EncoderLayer,
    MaxPoolingConstruct,
    Predictor,
    get_k_q,
    get_mask,
    get_q_k,
    get_subsequent_mask,
    refer_points,
)


class Encoder(nn.Module):
    """
    An Encoder model with a self-attention mechanism for sequence modeling.

    The encoder is configurable to use different types of attention mechanisms and embeddings,
    with potential TVM optimizations for efficiency.

    Args:
        opt: A configuration object containing model hyperparameters and settings.
    """

    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.d_model = opt.d_model
        self.model_type = opt.model
        self.window_size = opt.window_size
        self.truncate = opt.truncate
        if opt.decoder == "attention":
            self.mask, self.all_size = get_mask(
                opt.input_size, opt.window_size, opt.inner_size, opt.device
            )
        else:
            self.mask, self.all_size = get_mask(
                opt.input_size + 1, opt.window_size, opt.inner_size, opt.device
            )
        self.decoder_type = opt.decoder
        if opt.decoder == "FC":
            self.indexes = refer_points(self.all_size, opt.window_size, opt.device)

        if opt.use_tvm:
            assert (
                len(set(self.window_size)) == 1
            ), "Only constant window size is supported."
            padding = 1 if opt.decoder == "FC" else 0
            q_k_mask = get_q_k(
                opt.input_size + padding, opt.inner_size, opt.window_size[0], opt.device
            )
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        opt.d_model,
                        opt.d_inner_hid,
                        opt.n_head,
                        opt.d_k,
                        opt.d_v,
                        dropout=opt.dropout,
                        normalize_before=False,
                        use_tvm=True,
                        q_k_mask=q_k_mask,
                        k_q_mask=k_q_mask,
                    )
                    for i in range(opt.n_layer)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        opt.d_model,
                        opt.d_inner_hid,
                        opt.n_head,
                        opt.d_k,
                        opt.d_v,
                        dropout=opt.dropout,
                        normalize_before=False,
                    )
                    for i in range(opt.n_layer)
                ]
            )

        if opt.embed_type == "CustomEmbedding":
            self.enc_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)
            # self.enc_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        else:
            self.enc_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)

        self.conv_layers = eval(opt.CSCM)(
            opt.d_model, opt.window_size, opt.d_bottleneck
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder.

        Args:
            x_enc (torch.Tensor): The input tensor containing the encoded sequence data.
            x_mark_enc (torch.Tensor): The input tensor containing additional markers or features.

        Returns:
            torch.Tensor: The output tensor representing the encoded sequence.
        """
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        if self.decoder_type == "FC":
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(
                seq_enc.device
            )
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == "attention" and self.truncate:
            seq_enc = seq_enc[:, : self.all_size[0]]

        return seq_enc


class Model(nn.Module):
    """
    A sequence-to-sequence model with attention mechanism.

    This model integrates an encoder and, depending on the configuration, a decoder,
    to predict sequences. It supports various types of attention and prediction mechanisms.

    Args:
        opt: A configuration object containing model hyperparameters and settings.
    """

    def __init__(self, opt):
        super(Model, self).__init__()
        self.predict_step = opt.predict_step
        self.d_model = opt.d_model
        self.input_size = opt.input_size
        self.decoder_type = opt.decoder
        self.channels = opt.enc_in

        self.encoder = Encoder(opt)
        if opt.decoder == "attention":
            mask = get_subsequent_mask(
                opt.input_size, opt.window_size, opt.predict_step, opt.truncate
            )
            self.decoder = Decoder(opt, mask)
            self.predictor = Predictor(opt.d_model, opt.enc_in)
        elif opt.decoder == "FC":
            self.predictor = Predictor(4 * opt.d_model, opt.predict_step * opt.enc_in)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        pretrain: bool,
    ) -> torch.Tensor:
        """
        Forward pass of the Model.

        Args:
            x_enc (torch.Tensor): The input tensor for the encoder.
            x_mark_enc (torch.Tensor): Additional markers or features for the encoder.
            x_dec (torch.Tensor): The input tensor for the decoder (if used).
            x_mark_dec (torch.Tensor): Additional markers or features for the decoder (if used).
            pretrain (bool): Indicates whether the model is in pretraining mode.

        Returns:
            torch.Tensor: The output tensor of the model representing predictions.
        """
        if self.decoder_type == "attention":
            enc_output = self.encoder(x_enc, x_mark_enc)
            dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)

            if pretrain:
                dec_enc = torch.cat([enc_output[:, : self.input_size], dec_enc], dim=1)
                pred = self.predictor(dec_enc)
            else:
                pred = self.predictor(dec_enc)
        elif self.decoder_type == "FC":
            enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
            pred = self.predictor(enc_output).view(
                enc_output.size(0), self.predict_step, -1
            )

        return pred
