#!/usr/bin/env python
# coding: utf-8

"""Auxiliary layers for Transformer models."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

from typing import Tuple

import torch
from torch import nn


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


class MyLayerNorm(nn.Module):
    """A custom layer normalization module designed specifically for the seasonal part
    of a models. This layer normalization subtracts the mean of each feature across
    the temporal dimension from itself.

    Args:
        channels (int): The number of channels in the input feature map.
    """

    def __init__(self, channels: int):
        super(MyLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MyLayerNorm layer.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, sequence_length, channels).

        Returns:
            torch.Tensor: The normalized tensor.
        """
        x_hat = self.layer_norm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class MovingAvg(nn.Module):
    """A moving average block designed to highlight the trend in time series data.

    Args:
        kernel_size (int): The size of the moving window, determining how many values
                           are considered for each average calculation.
        stride (int): The stride of the pooling operation, determining how far the
                      pooling window moves for each calculation.
    """

    def __init__(self, kernel_size: int, stride: int):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MovingAvg layer.

        Args:
            x (torch.Tensor): The input tensor representing the time series, with shape
                          (batch_size, sequence_length, features).

        Returns:
            torch.Tensor: The smoothed time series tensor.
        """
        # Padding on both ends of the time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # Applying average pooling
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDeComp(nn.Module):
    """A series decomposition block used for decomposing time series data into its
    trend (moving average component) and residual components.

    Args:
        kernel_size (int): The size of the moving window for the moving average.
    """

    def __init__(self, kernel_size: int):
        super(SeriesDeComp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the SeriesDeComp layer.

        Args:
            x (torch.Tensor): The input time series tensor with shape (batch_size, sequence_length, features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the residual component and the moving average component of the time series.
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class SeriesDeCompMulti(nn.Module):
    """A series decomposition block for decomposing time series data into its trend and residual components,
    using multiple moving averages with different kernel sizes. This allows capturing trends at various scales.

    Args:
        kernel_size (List[int]): A list of kernel sizes for the moving averages.
    """

    def __init__(self, kernel_size: list):
        super(SeriesDeCompMulti, self).__init__()
        self.moving_avg = nn.ModuleList(
            [MovingAvg(kernel, stride=1) for kernel in kernel_size]
        )
        self.layer = nn.Linear(1, len(kernel_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SeriesDeCompMulti layer.

        Args:
            x (torch.Tensor): The input time series tensor with shape (batch_size, sequence_length, features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the residual component and the combined moving average component of the time series.
        """
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))

        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(
            moving_mean * nn.Softmax(dim=-1)(self.layer(x.unsqueeze(-1))), dim=-1
        )
        res = x - moving_mean
        return res, moving_mean
