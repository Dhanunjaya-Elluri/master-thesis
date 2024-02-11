#!/usr/bin/env python
# coding: utf-8

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import numpy as np
import pytest

from tqts.quantizer.paa import PAA


@pytest.fixture
def time_series_data():
    return np.arange(10, dtype=float)


@pytest.fixture
def paa_segments():
    return np.array([0.5, 2.5, 4.5])


# Test that the PAA class initializes correctly with valid inputs.
def test_paa_init():
    # Test initialization with window_size
    paa = PAA(window_size=3)
    assert paa.window_size == 3
    assert paa.n_windows is None

    # Test initialization with n_windows
    paa = PAA(n_windows=4)
    assert paa.n_windows == 4
    assert paa.window_size is None

    # Test initialization with both set (should raise ValueError)
    with pytest.raises(ValueError):
        PAA(window_size=3, n_windows=3)

    # Test initialization with neither set (should raise ValueError)
    with pytest.raises(ValueError):
        PAA()


# Test fitting method with window_size set.
def test_paa_fit_with_window_size(time_series_data):
    paa = PAA(window_size=2)
    paa.fit(time_series_data)
    assert paa.n_windows == 5  # Should compute correct number of windows.

    # Test with a window size too large
    paa_large_window = PAA(window_size=15)
    with pytest.raises(ValueError):
        paa_large_window.fit(time_series_data)


# Test fitting method with n_windows set.
def test_paa_fit_with_n_windows(time_series_data):
    paa = PAA(n_windows=5)
    paa.fit(time_series_data)
    assert paa.window_size == 2  # Should compute correct window size.

    # Test with too many windows
    paa_many_windows = PAA(n_windows=15)
    with pytest.raises(ValueError):
        paa_many_windows.fit(time_series_data)


# Test the transform method.
def test_paa_transform(time_series_data):
    paa = PAA(window_size=2)
    paa.fit(time_series_data)
    transformed = paa.transform(time_series_data)
    expected_result = np.array([0.5, 2.5, 4.5, 6.5, 8.5])  # The mean of each window.
    assert np.array_equal(transformed, expected_result)

    # Try to transform without fitting first.
    paa_not_fitted = PAA(window_size=3)
    with pytest.raises(ValueError):
        paa_not_fitted.transform(time_series_data)


# Test the inverse_transform method.
def test_paa_inverse_transform(paa_segments):
    paa = PAA(window_size=2)
    paa.fit(np.arange(6, dtype=float))
    reconstructed = paa.inverse_transform(paa_segments)
    expected_result = np.array(
        [0.5, 0.5, 2.5, 2.5, 4.5, 4.5]
    )  # The reconstructed time series.
    assert np.array_equal(reconstructed, expected_result)

    # Try to inverse transform without fitting first.
    paa_not_fitted = PAA(window_size=2)
    with pytest.raises(ValueError):
        paa_not_fitted.inverse_transform(paa_segments)
