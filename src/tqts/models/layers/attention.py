#!/usr/bin/env python
# coding: utf-8

"""Attention family for TQTS Model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"


import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from tqts.models.layers.masking import TriangularCasualMask, ProbMask


class FullAttention(nn.Module):
    """Full Attention module for the Informer Model."""

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale=None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        """Initialize the Full Attention module.

        Args:
            mask_flag (bool, optional): Mask flag. Defaults to True.
            factor (int, optional): Factor. Defaults to 5.
            scale ([type], optional): Scale. Defaults to None.
            attention_dropout (float, optional): Dropout probability. Defaults to 0.1.
            output_attention (bool, optional): Output attention flag. Defaults to False.
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> tuple:
        """Forward pass of the Full Attention module.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, H, seq_len, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, H, seq_len, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, H, seq_len, d_model).
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            tuple: Output tensor of shape (batch_size, H, seq_len, d_model) and attention tensor of shape (batch_size, H, seq_len, seq_len).
        """
        B, L, H, E = query.shape
        _, S, _, D = value.shape
        scale = self.scale or 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", query, key)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCasualMask(B, L, device=query.device).mask
            scores.masked_fill_(attn_mask, -np.inf)
        attn = torch.softmax(scores * scale, dim=-1)
        attn = self.dropout(attn)
        context = torch.einsum("bhls,bshd->blhd", attn, value)
        if self.output_attention:
            return context.contiguous(), attn
        else:
            return context.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: int = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ) -> None:
        """Initialize the Prob Attention module.

        Args:
            mask_flag (bool, optional): Mask flag. Defaults to True.
            factor (int, optional): Factor. Defaults to 5.
            scale ([type], optional): Scale. Defaults to None.
            attention_dropout (float, optional): Dropout probability. Defaults to 0.1.
            output_attention (bool, optional): Output attention flag. Defaults to False.
        """
        super(ProbAttention, self).__init__()
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(p=attention_dropout)
        self.output_attention = output_attention
        self.factor = factor
        self.scale = scale

    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, n_top: int):
        """Calculate the QK probability.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, H, seq_len, d_model).
            K (torch.Tensor): Key tensor of shape (batch_size, H, seq_len, d_model).
            sample_k (int): Sample K.
            n_top (int): Number of top.

        Returns:
            torch.Tensor: QK probability tensor of shape (batch_size, H, seq_len, sample_k).
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(
            -2
        )

        # find the top-k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q-K
        Q_reduce = Q[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V: torch.Tensor, L_Q: int):
        """Get the initial context.

        Args:
            V (torch.Tensor): Value tensor of shape (batch_size, H, seq_len, d_model).
            L_Q (int): Length of the query.

        Returns:
            torch.Tensor: Initial context tensor of shape (batch_size, H, L_Q, d_model).
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.sum(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert L_Q == L_V
            context = V.cumsum(dim=-2)
        return context

    def _update_context(
        self,
        context_in: torch.Tensor,
        V: torch.Tensor,
        scores: torch.Tensor,
        index: int,
        L_Q: int,
        attn_mask: torch.Tensor = None,
    ):
        """Update the context.

        Args:
            context_in (torch.Tensor): Input context tensor of shape (batch_size, H, L_Q, d_model).
            V (torch.Tensor): Value tensor of shape (batch_size, H, seq_len, d_model).
            scores (torch.Tensor): Scores tensor of shape (batch_size, H, L_Q, L_V).
            index (int): Index of the mask.
            L_Q (int): Length of the query.
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Updated context tensor of shape (batch_size, H, L_Q, d_model).
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attentions = (
                (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            )
            attentions[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attn
            return context_in, attentions
        else:
            return context_in, None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> tuple:
        """Forward pass of the Prob Attention module.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, H, seq_len, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, H, seq_len, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, H, seq_len, d_model).
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            tuple: Output tensor of shape (batch_size, H, seq_len, d_model)
            and attention tensor of shape (batch_size, H, seq_len, seq_len).
        """
        B, L_Q, H, D = query.shape
        _, L_K, _, _ = key.shape

        queries = query.transpose(2, 1)
        keys = key.transpose(2, 1)
        values = value.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype("int").item()
        u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1.0 / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys=None,
        d_values=None,
        mix: bool = False,
    ) -> None:
        """Initialize the Attention Layer module.

        Args:
            attention (nn.Module): Attention module.
            d_model (int): Embedding dimension.
            n_heads (int): Number of heads.
            d_keys ([type], optional): Keys dimension. Defaults to None.
            d_values ([type], optional): Values dimension. Defaults to None.
            mix (bool, optional): Mix flag. Defaults to False.
        """
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, n_heads * d_keys)
        self.key_projection = nn.Linear(d_model, n_heads * d_keys)
        self.value_projection = nn.Linear(d_model, n_heads * d_values)
        self.out_projection = nn.Linear(n_heads * d_values, d_model)
        self.mix = mix
        self.n_heads = n_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> tuple:
        """Forward pass of the Attention Layer module.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            attn_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, L, _ = query.shape
        _, S, _ = key.shape
        H = self.n_heads

        queries = self.query_projection(query).view(B, L, H, -1)
        keys = self.key_projection(key).view(B, S, H, -1)
        values = self.value_projection(value).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class LogSparseAttentionLayer(nn.Module):
    """A logarithmic sparse attention layer that applies a convolutional approach to
    query, key, and value projections in the attention mechanism. This layer is
    designed for efficient attention computations over long sequences.

    Args:
        attention (nn.Module): The attention mechanism to be used.
        d_model (int): Dimensionality of the models.
        n_heads (int): Number of attention heads.
        qk_ker (int): Kernel size for query and key convolutional projections.
        d_keys (Optional[int]): Dimensionality of keys. Defaults to d_model // n_heads.
        d_values (Optional[int]): Dimensionality of values. Defaults to d_model // n_heads.
        v_conv (bool): Whether to use convolution for projecting values. Defaults to False.
    """

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        qk_ker: int,
        d_keys: int = None,
        d_values: int = None,
        v_conv: bool = False,
        **_
    ):
        super(LogSparseAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.qk_ker = qk_ker
        self.query_projection = nn.Conv1d(d_model, d_keys * n_heads, self.qk_ker)
        self.key_projection = nn.Conv1d(d_model, d_keys * n_heads, self.qk_ker)
        self.v_conv = v_conv
        if v_conv:
            self.value_projection = nn.Conv1d(d_model, d_values * n_heads, self.qk_ker)
        else:
            self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
        **_
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the LogSparseAttentionLayer.

        Applies convolutional transformations to queries, keys, and values and then
        computes attention. The outputs are linearly projected before being returned.

        Args:
            queries (torch.Tensor): Input tensor for queries.
            keys (torch.Tensor): Input tensor for keys.
            values (torch.Tensor): Input tensor for values.
            attn_mask (torch.Tensor): Attention mask tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor after attention and linear projection, and attention weights.
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = nn.functional.pad(queries.permute(0, 2, 1), pad=(self.qk_ker - 1, 0))
        queries = self.query_projection(queries).permute(0, 2, 1).view(B, L, H, -1)

        keys = nn.functional.pad(keys.permute(0, 2, 1), pad=(self.qk_ker - 1, 0))
        keys = self.key_projection(keys).permute(0, 2, 1).view(B, S, H, -1)

        if self.v_conv:
            values = nn.functional.pad(
                values.permute(0, 2, 1), pad=(self.qk_ker - 1, 0)
            )
            values = self.value_projection(values).permute(0, 2, 1).view(B, S, H, -1)
        else:
            values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)

        out = out.view(B, L, -1)

        return self.out_projection(out), attn
