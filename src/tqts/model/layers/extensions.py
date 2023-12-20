#!/usr/bin/env python
# coding: utf-8

"""Individual Layers for the Transformer."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbeddings(nn.Module):
    """Input Embedding module for the Transformer."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """Initialize the Input Embedding module.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Embedding dimension.
        """
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

    def __init__(self, d_model: int, dropout: float) -> None:
        """Initialize the Positional Encoding module.

        Args:
            d_model (int): Embedding dimension.
            dropout (float): Dropout probability.
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Positional Encoding module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        seq_len = x.shape[1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / self.d_model)
        )
        pos_enc = torch.zeros(seq_len, self.d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).repeat(
            x.shape[0], 1, 1
        )  # to add a batch dimension (1, seq_len, d_model)
        self.register_buffer("pos_enc", pos_enc, persistent=False)
        return self.dropout(x + pos_enc.requires_grad_(False))


class AddAndNorm(nn.Module):
    """Add & Norm module for the Transformer."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """Initialize the Add & Norm module.

        Args:
            d_model (int): Embedding dimension.
            eps (float, optional): Epsilon value. Defaults to 1e-6.
        """
        super(AddAndNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Add & Norm module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    """Feed Forward module for the Transformer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """Initialize the Feed Forward module.

        Args:
            d_model (int): Embedding dimension.
            d_ff (int): Feedforward dimension.
            dropout (float): Dropout probability.
        """
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

    def __init__(self, dropout: float) -> None:
        """Initialize the Scaled Dot Product Attention module.

        Args:
            dropout (float): Dropout probability.
        """
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
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value)


class MultiHeadAttention(nn.Module):
    """Multi Head Attention module for the Transformer."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        """Initialize the Multi Head Attention module.

        Args:
            d_model (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.weights = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(4)]
        )
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
        batch_size = query.shape[0]  # query.size(0)
        seq_len = query.shape[1]
        query, key, value = [
            w(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            for w, x in zip(self.weights, (query, key, value))
        ]
        x = self.attention(query, key, value, mask)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.num_heads * self.d_k)
        )
        return self.weights[-1](x)


class ResidualConnection(nn.Module):
    """Residual Connection module for the Transformer."""

    def __init__(self, dropout: float, d_model: int) -> None:
        """Initialize the Residual Connection module.

        Args:
            dropout (float): Dropout probability.
            d_model (int): Embedding dimension.
        """
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = AddAndNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """Forward pass of the Residual Connection module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            sublayer (nn.Module): Sublayer module.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        return x + self.dropout(sublayer(self.norm(x)))


class LinearLayer(nn.Module):
    """Linear layer module for the Transformer."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """Initialize the Linear layer module.

        Args:
            d_model (int): Embedding dimension.
            vocab_size (int): Size of the vocabulary.
        """
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Linear layer module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size).
        """
        return self.linear(x)
