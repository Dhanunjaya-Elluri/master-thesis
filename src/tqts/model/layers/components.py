#!/usr/bin/env python
# coding: utf-8

"""Components for the Transformer."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbeddings(nn.Module):
    """Input Embedding module for the Transformer."""

    def __init__(self, vocab_size: int, d_model: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Input Embedding module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.embedding(x) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32)
        )


class PositionalEncoding(nn.Module):
    """Positional Encoding module for the Transformer."""

    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super(PositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pos_enc = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # Shape: (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).transpose(
            0, 1
        )  # to add a batch dimension (1, seq_len, d_model)
        self.register_buffer(
            "pos_enc", pos_enc
        )  # Register the positional encodings as a buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Positional Encoding module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        x = x + self.pos_enc[: x.size(0), :].requires_grad_(
            False
        )  # Set requires_grad to False to avoid computing gradients
        return self.dropout(x)


class AddNorm(nn.Module):
    """Add & Norm module for the Transformer."""

    def __init__(self, eps: float = 1e-6):
        super(AddNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Add & Norm module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.beta) + self.bias


class FeedForward(nn.Module):
    """Feed Forward module for the Transformer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Feed Forward module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot Product Attention module for the Transformer."""

    def __init__(self, dropout: float):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of the Scaled Dot Product Attention module.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value)


class MultiHeadAttention(nn.Module):
    """Multi Head Attention module for the Transformer."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.weights = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of the Multi Head Attention module.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        query, key, value = [
            w(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for w, x in zip(self.weights, (query, key, value))
        ]
        x = self.attention(query, key, value, mask=mask)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )
        return self.weights[-1](x)


class ResidualConnection(nn.Module):
    """Residual Connection module for the Transformer."""

    def __init__(self, dropout: float):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = AddNorm()

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """Forward pass of the Residual Connection module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            sublayer (nn.Module): Sublayer module.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        return x + self.dropout(sublayer(self.norm(x)))


class LinearLayer(nn.Module):
    """Linear layer module for the Transformer."""

    def __init__(self, d_model: int, vocab_size: int):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Linear layer module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, output_size).
        """
        return self.linear(x)
