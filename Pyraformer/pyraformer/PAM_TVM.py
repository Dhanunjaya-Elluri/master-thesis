#!/usr/bin/env python
# coding: utf-8

"""Pyramidal Attention for Pyraformer model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"


import math

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pyraformer.hierarchical_mm_tvm import graph_mm as graph_mm_tvm


class PyramidalAttention(nn.Module):
    """
    PyramidalAttention is a module implementing a pyramidal attention mechanism.

    This attention mechanism is a variant of the standard multi-head attention and is designed to capture
    hierarchical relationships in data. It uses custom graph-based matrix multiplication (graph_mm_tvm) to
    compute attention weights and context vectors.

    Args:
        n_head (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        d_k (int): Dimension of key/query vectors.
        d_v (int): Dimension of value vectors.
        dropout (float): Dropout rate.
        normalize_before (bool): Whether to normalize before computing attention.
        q_k_mask (Tensor): Query-key mask tensor.
        k_q_mask (Tensor): Key-query mask tensor.
    """

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float,
        normalize_before: bool,
        q_k_mask: Tensor,
        k_q_mask: Tensor,
    ) -> None:
        super(PyramidalAttention, self).__init__()
        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_k, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_k * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_fc = nn.Dropout(dropout)
        self.q_k_mask = q_k_mask
        self.k_q_mask = k_q_mask

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Forward pass of the PyramidalAttention.

        Args:
            hidden_states (Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            Tensor: Output tensor of the same shape as hidden_states.
        """
        residual = hidden_states

        hidden_states = hidden_states
        bsz, seq_len, _ = hidden_states.size()

        q = hidden_states
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)
        q /= math.sqrt(self.d_k)

        q = q.view(bsz, seq_len, self.n_head, self.d_k)
        k = k.view(bsz, seq_len, self.n_head, self.d_k)
        q = q.float().contiguous()
        k = k.float().contiguous()
        # attn_weights.size(): (batch_size, L, num_heads, 11)
        attn_weights = graph_mm_tvm(q, k, self.q_k_mask, self.k_q_mask, False, 0)
        attn_weights = self.dropout_attn(F.softmax(attn_weights, dim=-1))

        v = v.view(bsz, seq_len, self.n_head, self.d_k)
        v = v.float().contiguous()
        # is_t1_diagonaled=True
        attn = graph_mm_tvm(attn_weights, v, self.q_k_mask, self.k_q_mask, True, 0)
        attn = attn.reshape(bsz, seq_len, self.n_head * self.d_k).contiguous()
        context = self.dropout_fc(self.fc(attn))
        context += residual

        if not self.normalize_before:
            context = self.layer_norm(context)

        return context
