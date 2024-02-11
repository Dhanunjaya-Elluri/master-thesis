#!/usr/bin/env python
# coding: utf-8

"""Test module for KernelSAX."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import numpy as np
import pytest

from tqts.quantizer.kernel_sax import KernelSAX


@pytest.fixture
def sample_time_series():
    """Fixture to provide a sample time series."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def ksax_instance():
    """Fixture to provide a KernelSAX instance."""
    return KernelSAX(
        kernel="gaussian",
        n_alphabet=5,
        bandwidth=1,
        boundary_estimator="lloyd-max",
        epochs=10,
        random_state=42,
        paa_window_size=2,
    )


def test_fit(sample_time_series, ksax_instance):
    """Test the fit method."""
    alphabets = ksax_instance.fit(sample_time_series)
    assert (
        len(alphabets) == len(sample_time_series) // ksax_instance.paa_window_size
    ), f"Expected {len(sample_time_series) // ksax_instance.paa_window_size} alphabets, but got {len(alphabets)}."


def test_encode_with_lloyd_boundaries(sample_time_series, ksax_instance):
    """Test the encode_with_lloyd_boundaries method."""
    ksax_instance.fit(sample_time_series)
    assert (
        ksax_instance.assigned_codewords is not None
    ), f"Expected assigned_codewords to be not None, but got {ksax_instance.assigned_codewords}."


def test_decode_with_lloyd_boundaries(sample_time_series, ksax_instance):
    """Test the decode_with_lloyd_boundaries method."""
    ksax_instance.fit(sample_time_series)
    original_values = ksax_instance.decode_with_lloyd_boundaries()
    assert (
        len(original_values) == len(sample_time_series) // ksax_instance.paa_window_size
    ), f"Expected {len(sample_time_series) // ksax_instance.paa_window_size} original values, but got {len(original_values)}."
