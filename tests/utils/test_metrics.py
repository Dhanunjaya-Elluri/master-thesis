#!/usr/bin/env python
# coding: utf-8

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

"""Test module for metrics."""

import numpy as np
import pytest

from tqts.utils.metrics import CORR, MAE, MAPE, MSE, MSPE, RMSE, RSE, metric


@pytest.fixture
def pred_true_pairs():
    """Fixture to provide mock predicted and true values."""
    pred = np.array([2, 3, 4, 5])
    true = np.array([3, 4, 5, 6])
    return pred, true


@pytest.fixture
def perfect_pred_true_pairs():
    """Fixture to provide mock predicted and true values that are perfect (no error)."""
    pred = np.array([1, 2, 3])
    true = np.array([1, 2, 3])
    return pred, true


def test_RSE(pred_true_pairs):
    """Test the Relative Squared Error (RSE) metric."""
    pred, true = pred_true_pairs
    pred, true = pred_true_pairs
    result = RSE(pred, true)
    expected_result = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )
    assert np.isclose(
        result, expected_result
    ), f"RSE calculation is incorrect: {result} != {expected_result}"


def test_CORR(perfect_pred_true_pairs):
    """Test the CORR function with perfect prediction, expecting correlation of nearly 0.01."""
    pred, true = perfect_pred_true_pairs
    result = CORR(pred, true)
    assert np.isclose(
        result, 0.01, atol=1e-2
    ), f"CORR calculation is incorrect with perfect prediction: {result} != 0.01"


def test_MAE(pred_true_pairs):
    """Test the Mean Absolute Error function."""
    pred, true = pred_true_pairs
    result = MAE(pred, true)
    expected_result = np.mean(np.abs(pred - true))
    assert np.isclose(
        result, expected_result
    ), f"MAE calculation is incorrect: {result} != {expected_result}"


def test_MSE(pred_true_pairs):
    """Test the Mean Squared Error function."""
    pred, true = pred_true_pairs
    result = MSE(pred, true)
    expected_result = np.mean((pred - true) ** 2)
    assert np.isclose(
        result, expected_result
    ), f"MSE calculation is incorrect: {result} != {expected_result}"


def test_RMSE(pred_true_pairs):
    """Test the Root Mean Squared Error function."""
    pred, true = pred_true_pairs
    result = RMSE(pred, true)
    expected_result = np.sqrt(MSE(pred, true))
    assert np.isclose(
        result, expected_result
    ), f"RMSE calculation is incorrect: {result} != {expected_result}"


def test_MAPE(pred_true_pairs):
    """Test the Mean Absolute Percentage Error function."""
    pred, true = pred_true_pairs
    result = MAPE(pred, true)
    expected_result = np.mean(np.abs((pred - true) / true))
    assert np.isclose(
        result, expected_result
    ), f"MAPE calculation is incorrect: {result} != {expected_result}"


def test_MSPE(pred_true_pairs):
    """Test the Mean Squared Percentage Error function."""
    pred, true = pred_true_pairs
    result = MSPE(pred, true)
    expected_result = np.mean(np.square((pred - true) / true))
    assert np.isclose(
        result, expected_result
    ), f"MSPE calculation is incorrect: {result} != {expected_result}"


def test_metric(pred_true_pairs):
    """Test the metric function that aggregates multiple metrics."""
    pred, true = pred_true_pairs
    results = metric(pred, true)
    expected_results = (
        MAE(pred, true),
        MSE(pred, true),
        RMSE(pred, true),
        MAPE(pred, true),
        MSPE(pred, true),
        RSE(pred, true),
        CORR(pred, true),
    )
    assert all(
        np.isclose(r, e) for r, e in zip(results, expected_results)
    ), f"Metric calculation is incorrect: {results} != {expected_results}"
