#!/usr/bin/env python
# coding: utf-8

"""Embeddings for TQTS Model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

import torch
import torch.nn as nn


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
            bias=False,
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
    """Temporal Embedding for Informer models."""

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
    """Time Feature Embedding for Informer models."""

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
    """Data Embedding module for the Informer Model.

    This module combines value, temporal, and positional embeddings to provide a comprehensive
    representation of input time series data.

    Args:
        c_in (int): The number of input channels.
        d_model (int): The dimensionality of the output embeddings.
        embed_type (str): The type of temporal embedding ('fixed', 'timeF', etc.).
        freq (str): The frequency of the time series data ('h' for hourly, etc.).
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super(DataEmbedding, self).__init__()

        self.token_embed = TokenEmbedding(c_in, d_model)
        self.positional_embed = PositionalEncoding(d_model)

        self.temporal_embed = (
            TemporalEmbedding(d_model, embed_type, freq)
            if embed_type == "fixed"
            else TimeFeatureEmbedding(d_model, embed_type, freq)
        )
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


class LogSparseTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3, spatial=False):
        super(LogSparseTokenEmbedding, self).__init__()
        assert torch.__version__ >= "1.5.0"
        padding = kernel_size - 1
        self.spatial = spatial
        self.d_model = d_model
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="zeros",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        B, L, d = x.shape[:3]
        if self.spatial:
            x = x.permute(0, 3, 1, 2).reshape(-1, L, d)
            x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)[:, :L, :]
            x = x.reshape(B, -1, L, self.d_model).permute(0, 2, 3, 1)
        else:
            x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)[:, :L, :]
        return x


class LogSparseTemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(LogSparseTemporalEmbedding, self).__init__()

        minute_size = 6
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t" or freq == "10min":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class LogSparseTimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(LogSparseTimeFeatureEmbedding, self).__init__()

        freq_map = {
            "h": 4,
            "t": 5,
            "s": 6,
            "m": 1,
            "a": 1,
            "w": 2,
            "d": 3,
            "b": 3,
            "10min": 5,
        }
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class LogSparseDataEmbedding(nn.Module):
    """A data embedding module for time series data, integrating value, positional, and
    temporal embeddings.

    This module is designed to process time series data by embedding input features,
    temporal information, and positional information, making it suitable for models
    handling sequential data.

    Args:
        c_in (int): Number of input channels.
        d_model (int): Dimensionality of the models.
        embed_type (str): Type of temporal embedding. Options: 'fixed', 'timeF'. Defaults to 'fixed'.
        freq (str): Frequency of the data. Defaults to 'h'.
        dropout (float): Dropout rate. Defaults to 0.1.
        kernel_size (int): Kernel size for the value embedding. Defaults to 3.
        spatial (bool): Flag to enable spatial embedding. Defaults to False.
        temp_embed (bool): Flag to enable temporal embedding. Defaults to True.
        d_pos (Optional[int]): Dimensionality of the positional embedding. If not provided, defaults to d_model.
        pos_embed (bool): Flag to enable positional embedding. Defaults to True.
    """

    def __init__(
        self,
        c_in,
        d_model,
        embed_type="fixed",
        freq="h",
        dropout=0.1,
        kernel_size=3,
        spatial=False,
        temp_embed=True,
        d_pos=None,
        pos_embed=True,
    ):
        super(LogSparseDataEmbedding, self).__init__()

        self.value_embedding = LogSparseTokenEmbedding(
            c_in=c_in, d_model=d_model, kernel_size=kernel_size, spatial=spatial
        )
        self.d_model = d_model
        if d_pos is None:
            self.d_pos = d_model
        else:
            self.d_pos = d_pos
        self.position_embedding = (
            PositionalEncoding(d_model=self.d_pos) if pos_embed else None
        )
        if temp_embed:
            self.temporal_embedding = (
                LogSparseTemporalEmbedding(
                    d_model=d_model, embed_type=embed_type, freq=freq
                )
                if embed_type != "timeF"
                else LogSparseTimeFeatureEmbedding(
                    d_model=d_model, embed_type=embed_type, freq=freq
                )
            )
        else:
            self.temporal_embedding = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, n_node=None):
        val_embed = self.value_embedding(x)
        temp_embed = (
            self.temporal_embedding(x_mark)
            if self.temporal_embedding is not None
            else None
        )
        pos_embed = (
            self.position_embedding(x) if self.position_embedding is not None else None
        )
        if self.d_pos != self.d_model and pos_embed is not None:
            pos_embed = pos_embed.repeat_interleave(2, dim=-1)
        if temp_embed is not None:
            if not (
                len(val_embed.shape) == len(temp_embed.shape)
            ):  # == len(pos_embed.shape)
                temp_embed = torch.unsqueeze(temp_embed, -1)
                pos_embed = (
                    torch.unsqueeze(pos_embed, -1) if pos_embed is not None else None
                )
        if n_node is not None and temp_embed is not None:
            temp_embed = torch.repeat_interleave(temp_embed, n_node, 0)
        if pos_embed is not None:
            x = (
                val_embed + temp_embed + pos_embed
                if temp_embed is not None
                else val_embed + pos_embed
            )
        else:
            x = val_embed + temp_embed if temp_embed is not None else val_embed
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """
    DataEmbedding_wo_pos is a module for time series data embedding without positional information.

    It combines value and temporal embeddings to represent input time series data, excluding
    positional embeddings to possibly cater to datasets where position might not be as relevant.

    Attributes:
        value_embedding (TokenEmbedding): Embeds the input values.
        temporal_embedding (TemporalEmbedding or TimeFeatureEmbedding): Embeds the time-related features.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ):
        """
        Initializes the DataEmbedding_wo_pos module.

        Args:
            c_in (int): The number of input channels.
            d_model (int): The dimensionality of the output embeddings.
            embed_type (str): The type of temporal embedding ('fixed', 'timeF', etc.).
            freq (str): The frequency of the time series data ('h' for hourly, etc.).
            dropout (float): The dropout rate.
        """
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        # Choose the correct temporal embedding based on the embed_type
        if embed_type == "timeF":
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, frequency=freq
            )
        else:
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model, embed_type=embed_type, frequency=freq
            )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Forward pass of the DataEmbedding_wo_pos module.

        Args:
            x (Tensor): The input data.
            x_mark (Tensor): The temporal markers associated with the input data.

        Returns:
            Tensor: The resulting embedded output.
        """
        # Combining value and temporal embeddings
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):
    """
    DataEmbedding_wo_pos_temp is a module for time series data embedding without positional and temporal information.

    It solely relies on value embeddings to represent input time series data, excluding
    positional and temporal embeddings, catering to datasets where these might not be relevant.

    Attributes:
        value_embedding (TokenEmbedding): Embeds the input values.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ):
        """
        Initializes the DataEmbedding_wo_pos_temp module.

        Args:
            c_in (int): The number of input channels.
            d_model (int): The dimensionality of the output embeddings.
            embed_type (str): The type of embedding ('fixed', 'timeF', etc. - not used in this class).
            freq (str): The frequency of the time series data ('h' for hourly, etc. - not used in this class).
            dropout (float): The dropout rate.
        """
        super(DataEmbedding_wo_pos_temp, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Forward pass of the DataEmbedding_wo_pos_temp module.

        Args:
            x (Tensor): The input data.
            x_mark (Tensor): The temporal markers associated with the input data (not used in this class).

        Returns:
            Tensor: The resulting embedded output.
        """
        x = self.value_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):
    """
    DataEmbedding_wo_temp is a module for time series data embedding without temporal information.

    It combines value and positional embeddings to represent input time series data, excluding
    temporal embeddings to cater to datasets where time might not be as relevant.

    Attributes:
        value_embedding (TokenEmbedding): Embeds the input values.
        position_embedding (PositionalEmbedding): Provides positional context to the inputs.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ):
        """
        Initializes the DataEmbedding_wo_temp module.

        Args:
            c_in (int): The number of input channels.
            d_model (int): The dimensionality of the output embeddings.
            embed_type (str): The type of embedding ('fixed', 'timeF', etc. - not used in this class).
            freq (str): The frequency of the time series data ('h' for hourly, etc. - not used in this class).
            dropout (float): The dropout rate.
        """
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Forward pass of the DataEmbedding_wo_temp module.

        Args:
            x (Tensor): The input data.
            x_mark (Tensor): The temporal markers associated with the input data (not used in this class).

        Returns:
            Tensor: The resulting embedded output.
        """
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
