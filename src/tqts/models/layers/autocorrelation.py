#!/usr/bin/env python
# coding: utf-8

"""Autocorrelation Layers for AutoFormer models."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

import math

import torch
import torch.nn as nn


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism for capturing period-based dependencies and time-delayed aggregations.
    This block can replace the self-attention family mechanism seamlessly.

    Attributes:
        mask_flag (bool): If True, masking is applied.
        factor (int): Determines the number of top elements to select.
        scale (float): Scaling factor for the attention scores.
        attention_dropout (float): Dropout rate for attention weights.
        output_attention (bool): If True, outputs the attention weights.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 1,
        scale: float = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super(AutoCorrelation, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(
        self, values: torch.Tensor, corr: torch.Tensor
    ) -> torch.Tensor:
        """
        SpeedUp version of Autocorrelation for the training phase.
        It's a batch-normalization style design.

        Args:
            values (torch.Tensor): The values tensor.
            corr (torch.Tensor): The correlation tensor.

        Returns:
            torch.Tensor: The aggregated tensor after applying time delay.
        """
        head, channel, length = values.shape[1:4]
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg += pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                1
            ).repeat(1, head, channel, length)
        return delays_agg

    def time_delay_agg_inference(
        self, values: torch.Tensor, corr: torch.Tensor
    ) -> torch.Tensor:
        """
        SpeedUp version of Autocorrelation for the inference phase.
        It's a batch-normalization style design.

        Args:
            values (torch.Tensor): The values tensor.
            corr (torch.Tensor): The correlation tensor.

        Returns:
            torch.Tensor: The aggregated tensor after applying time delay.
        """
        batch, head, channel, length = values.shape[0:4]
        init_index = (
            torch.arange(length)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
            .cuda()
        )
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                1
            ).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg += pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                1
            ).repeat(1, head, channel, length)
        return delays_agg

    def time_delay_agg_full(
        self, values: torch.Tensor, corr: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard version of Autocorrelation.

        Args:
            values (torch.Tensor): The values tensor.
            corr (torch.Tensor): The correlation tensor.

        Returns:
            torch.Tensor: The aggregated tensor after applying time delay.
        """
        batch, head, channel, length = values.shape
        init_index = (
            torch.arange(length)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
            .cuda()
        )
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg += pattern * tmp_corr[..., i].unsqueeze(-1)
        return delays_agg

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
    ):
        """
        Forward pass of the AutoCorrelation block.

        Args:
            queries (torch.Tensor): The query tensor.
            keys (torch.Tensor): The key tensor.
            values (torch.Tensor): The value tensor.
            attn_mask (torch.Tensor): The attention mask tensor.

        Returns:
            tuple: A tuple containing the output values and correlation tensor.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, : (L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        if self.training:
            V = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        else:
            return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):
    """
    A layer that applies the AutoCorrelation mechanism to the input.

    Attributes:
        correlation (nn.Module): The autocorrelation module to use.
        d_model (int): The number of expected features in the input.
        n_heads (int): The number of heads in the multiheadattention models.
        d_keys (int, optional): Size of the key projections per attention head. Defaults to d_model // n_heads.
        d_values (int, optional): Size of the value projections per attention head. Defaults to d_model // n_heads.
    """

    def __init__(
        self,
        correlation: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: int = None,
        d_values: int = None,
    ):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> tuple:
        """
        Forward pass of the AutoCorrelation layer.

        Args:
            queries (torch.Tensor): The query tensor.
            keys (torch.Tensor): The key tensor.
            values (torch.Tensor): The value tensor.
            attn_mask (torch.Tensor): The attention mask tensor.

        Returns:
            tuple: A tuple containing the output tensor and attention weights.
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
