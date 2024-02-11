#!/usr/bin/env python
# coding: utf-8

"""Metric functions for evaluating model performance."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

from typing import Tuple, Union

import numpy as np
import torch


def MAE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Computes the Mean Absolute Error between predictions and true values.

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: The Mean Absolute Error.
    """
    return np.mean(np.abs(pred - true))


def MSE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Computes the Mean Squared Error between predictions and true values.

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: The Mean Squared Error.
    """
    return np.mean((pred - true) ** 2)


def RMSE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error between predictions and true values.

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: The Root Mean Squared Error.
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Computes the Mean Absolute Percentage Error between predictions and true values.

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: The Mean Absolute Percentage Error.
    """
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Computes the Mean Squared Percentage Error between predictions and true values.

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: The Mean Squared Percentage Error.
    """
    return np.mean(np.square((pred - true) / true))


def metric(
    pred: np.ndarray, true: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """
    Computes multiple error metrics between predictions and true values.

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        Tuple[float, float, float, float, float]: Tuple containing MAE, MSE, RMSE, MAPE, MSPE.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


class StandardScaler:
    """
    A standard scaler for normalizing data.

    This class computes and applies mean normalization and standard deviation scaling.
    """

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data: np.ndarray):
        """
        Fits the scaler on the data.

        Args:
            data (np.ndarray): The data to fit the scaler on.
        """
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Transforms the data using the fitted mean and standard deviation.

        Args:
            data (Union[np.ndarray, torch.Tensor]): The data to transform.

        Returns:
            Union[np.ndarray, torch.Tensor]: Transformed data.
        """
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data - mean) / std

    def inverse_transform(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Inversely transforms the data using the fitted mean and standard deviation.

        Args:
            data (Union[np.ndarray, torch.Tensor]): The data to inversely transform.

        Returns:
            Union[np.ndarray, torch.Tensor]: Inversely transformed data.
        """
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data * std) + mean


class TopkMSELoss(torch.nn.Module):
    """
    A custom loss module that computes the top-k Mean Squared Error.

    Args:
        topk (int): The number of highest errors to consider.
    """

    def __init__(self, topk: int):
        super().__init__()
        self.topk = topk
        self.criterion = torch.nn.MSELoss(reduction="none")

    def forward(self, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss computation.

        Args:
            output (torch.Tensor): Predicted values.
            label (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: The computed top-k Mean Squared Error.
        """
        losses = self.criterion(output, label).mean(2).mean(1)
        losses = torch.topk(losses, self.topk)[0]

        return losses


class SingleStepLoss(torch.nn.Module):
    """
    Computes top-k log-likelihood and mean squared error for single-step predictions.

    Args:
        ignore_zero (bool): Flag to ignore zero values in labels during loss computation.
    """

    def __init__(self, ignore_zero: bool):
        super().__init__()
        self.ignore_zero = ignore_zero

    def forward(
        self, mu: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor, topk: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the loss computation.

        Args:
            mu (torch.Tensor): Mean predictions.
            sigma (torch.Tensor): Standard deviation of predictions.
            labels (torch.Tensor): Ground truth values.
            topk (int): The number of highest errors to consider.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Log-likelihood and mean squared error.
        """
        if self.ignore_zero:
            indexes = labels != 0
        else:
            indexes = labels >= 0

        distribution = torch.distributions.normal.Normal(mu[indexes], sigma[indexes])
        likelihood = -distribution.log_prob(labels[indexes])

        diff = labels[indexes] - mu[indexes]
        se = diff * diff

        if 0 < topk < len(likelihood):
            likelihood = torch.topk(likelihood, topk)[0]
            se = torch.topk(se, topk)[0]

        return likelihood, se


def AE_loss(mu: torch.Tensor, labels: torch.Tensor, ignore_zero: bool) -> torch.Tensor:
    """
    Computes the absolute error between predictions and true values, optionally ignoring zeros.

    Args:
        mu (torch.Tensor): Predicted values.
        labels (torch.Tensor): Ground truth values.
        ignore_zero (bool): Flag to ignore zero values in labels during loss computation.

    Returns:
        torch.Tensor: The computed absolute error.
    """
    if ignore_zero:
        indexes = labels != 0
    else:
        indexes = labels >= 0

    ae = torch.abs(labels[indexes] - mu[indexes])
    return ae
