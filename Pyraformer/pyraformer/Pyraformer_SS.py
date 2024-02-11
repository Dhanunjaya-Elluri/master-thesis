#!/usr/bin/env python
# coding: utf-8

""" section for Pyraformer model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

from typing import Tuple

import torch
import torch.nn as nn

from Pyraformer.pyraformer.embed import SingleStepEmbedding
from Pyraformer.pyraformer.Layers import (
    BottleneckConstruct,
    EncoderLayer,
    Predictor,
    get_k_q,
    get_mask,
    get_q_k,
    refer_points,
)


class Encoder(nn.Module):
    """
    An Encoder model with a self-attention mechanism.

    This encoder is a part of a sequence modeling architecture, utilizing self-attention
    and potentially TVM optimizations for efficient processing.

    Args:
        opt: A configuration object containing model hyperparameters and settings.
    """

    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.d_model = opt.d_model
        self.window_size = opt.window_size
        self.num_heads = opt.n_head
        self.mask, self.all_size = get_mask(
            opt.input_size, opt.window_size, opt.inner_size, opt.device
        )
        self.indexes = refer_points(self.all_size, opt.window_size, opt.device)

        if opt.use_tvm:
            assert (
                len(set(self.window_size)) == 1
            ), "Only constant window size is supported."
            q_k_mask = get_q_k(
                opt.input_size, opt.inner_size, opt.window_size[0], opt.device
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

        self.embedding = SingleStepEmbedding(
            opt.covariate_size, opt.num_seq, opt.d_model, opt.input_size, opt.device
        )

        self.conv_layers = BottleneckConstruct(opt.d_model, opt.window_size, opt.d_k)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder.

        Args:
            sequence (torch.Tensor): Input tensor of sequence data.

        Returns:
            torch.Tensor: Output tensor after encoding the sequence.
        """
        seq_enc = self.embedding(sequence)
        mask = self.mask.repeat(len(seq_enc), self.num_heads, 1, 1).to(sequence.device)

        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(
            seq_enc.device
        )
        indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
        all_enc = torch.gather(seq_enc, 1, indexes)
        all_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)

        return all_enc


class Model(nn.Module):
    """
    A complete model integrating an encoder with prediction layers.

    This model encapsulates an encoder for sequence data and additional layers to
    predict certain properties (like mean and variance in this context) from the
    encoded representations.

    Args:
        opt: A configuration object containing model hyperparameters and settings.
    """

    def __init__(self, opt):
        super(Model, self).__init__()
        self.encoder = Encoder(opt)

        # convert hidden vectors into two scalar
        self.mean_hidden = Predictor(4 * opt.d_model, 1)
        self.var_hidden = Predictor(4 * opt.d_model, 1)

        self.softplus = nn.Softplus()

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Model.

        Args:
            data (torch.Tensor): Input tensor of sequence data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing mean and variance predictions.
        """
        enc_output = self.encoder(data)

        mean_pre = self.mean_hidden(enc_output)
        var_hid = self.var_hidden(enc_output)
        var_pre = self.softplus(var_hid)
        mean_pre = self.softplus(mean_pre)

        return mean_pre.squeeze(2), var_pre.squeeze(2)

    def test(self, data: torch.Tensor, v: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Test method for the Model.

        Args:
            data (torch.Tensor): Input tensor of sequence data.
            v (float): A scaling factor for the predictions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Scaled mean and variance predictions.
        """
        mu, sigma = self(data)

        sample_mu = mu[:, -1] * v
        sample_sigma = sigma[:, -1] * v
        return sample_mu, sample_sigma
