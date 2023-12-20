#!/usr/bin/env python
# coding: utf-8

"""Extensions for Informer Model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import math

import numpy as np
import torch
import torch.nn as nn


class TriangularCasualMask:
    """Triangular Casual Mask module for the Informer Model."""

    def __init__(self, B: int, L: int, device: str = "cpu"):
        """Initialize the Triangular Casual Mask module.

        Args:
            B (int): Batch size.
            L (int): Sequence length.
            device (torch.device): Device to use.
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        """Get the mask tensor.

        Returns:
            torch.Tensor: Mask tensor of shape (B, 1, L, L).
        """
        return self._mask


class ProbMask:
    def __init__(
        self,
        B: int,
        H: int,
        L: int,
        index: int,
        scores: torch.Tensor,
        device: str = "cpu",
    ):
        """Initialize the Prob Mask module.

        Args:
            B (int): Batch size.
            H (int): Number of heads.
            L (int): Sequence length.
            index (int): Index of the mask.
            scores (torch.Tensor): Scores tensor of shape (B, H, L, L).
            device (torch.device): Device to use.
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """Get the mask tensor.

        Returns:
            torch.Tensor: Mask tensor of shape (B, H, L, L).
        """
        return self._mask


class ConvLayer(nn.Module):
    """Convolutional Layer module for the Informer Model."""

    def __init__(
        self,
        in_channels: int,
    ) -> None:
        """Initialize the Convolutional Layer module.

        Args:
            in_channels (int): Number of input channels.
        """
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.activation = nn.ELU()
        self.norm = nn.BatchNorm1d(in_channels)
        self.max_pool = nn.MaxPool1d(kernel_size=3, padding=1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Convolutional Layer module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, seq_len).
        """
        x = self.conv(x.permute(0, 2, 1))
        x = self.activation(self.norm(x))
        x = self.max_pool(x)
        return x.transpose(1, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len: int = 5000):
        """Initialize the Positional Encoding module.

        Args:
            d_model (int): Embedding dimension.
            max_len (int, optional): Maximum length of the input sequence. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        pos_enc = torch.zeros(max_len, d_model).float()
        pos_enc.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Positional Encoding module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.pos_enc[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """Token Embedding module for the Informer Model."""

    def __init__(self, c_in: int, d_model: int) -> None:
        """Initialize the Token Embedding module.

        Args:
            c_in (int): Number of input channels.
            d_model (int): Embedding dimension.
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Token Embedding module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, c_in, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, d_model, seq_len).
        """
        x = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """Fixed Embedding module for the Informer Model."""

    def __int__(self, c_in: int, d_model: int) -> None:
        """Initialize the Fixed Embedding module.

        Args:
            c_in (int): Number of input channels.
            d_model (int): Embedding dimension.
        """
        super(FixedEmbedding, self).__init__()
        weight = torch.zeros(c_in, d_model).float()
        weight.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)
        self.embed = nn.Embedding(c_in, d_model)
        self.embed.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fixed Embedding module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.embed(x).detach()


class TemporalEmbedding(nn.Module):
    """Temporal Embedding for Informer model."""

    def __init__(
        self, d_model: int, embed_type: str = "fixed", frequency: str = "h"
    ) -> None:
        """Initialize the Temporal Embedding module.

        Args:
            d_model (int): Embedding dimension.
            embed_type (str, optional): Type of embedding. Defaults to "fixed".
            frequency (str, optional): Frequency of the input data. Defaults to "h".
        """
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        embedding = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if frequency == "t":
            self.minute_embed = embedding(minute_size, d_model)
        self.hour_embed = embedding(hour_size, d_model)
        self.weekday_embed = embedding(weekday_size, d_model)
        self.day_embed = embedding(day_size, d_model)
        self.month_embed = embedding(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Temporal Embedding module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return minute_x + hour_x + weekday_x + day_x + month_x


class TimeFeatureEmbedding(nn.Module):
    """Time Feature Embedding for Informer model."""

    def __init__(
        self, d_model: int, embed_type: str = "timeF", frequency: str = "h"
    ) -> None:
        super(TimeFeatureEmbedding, self).__init__()
        frequency_mapping = {
            "h": 4,
            "t": 5,
            "s": 6,
            "m": 1,
            "a": 1,
            "w": 2,
            "d": 3,
            "b": 3,
        }
        d_input = frequency_mapping[frequency]
        self.embed = nn.Linear(d_input, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Time Feature Embedding module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_input).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """Data Embedding module for the Informer Model."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        """Initialize the Data Embedding module.

        Args:
            c_in (int): Number of input channels.
            d_model (int): Embedding dimension.
            embed_type (str, optional): Type of embedding. Defaults to "timeF".
            freq (str, optional): Frequency of the input data. Defaults to "h".
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(DataEmbedding, self).__init__()
        self.token_embed = TokenEmbedding(c_in, d_model)
        self.temporal_embed = (
            TemporalEmbedding(d_model, embed_type, freq)
            if embed_type == "fixed"
            else TimeFeatureEmbedding(d_model, embed_type, freq)
        )
        self.positional_embed = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark) -> torch.Tensor:
        """Forward pass of the Data Embedding module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, c_in, seq_len).
            x_mark (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_input).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.token_embed(x) + self.positional_embed(x) + self.temporal_embed(x_mark)
        return self.dropout(x)


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
            tuple: Output tensor of shape (batch_size, H, seq_len, d_model) and attention tensor of shape (batch_size, H, seq_len, seq_len).
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
