#!/usr/bin/env python
# coding: utf-8

"""Test module for KernelSAX."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import pytest
import numpy as np
from tqts.quantizer.kernel_sax import KernelSAX


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def ksax_instance():
    """Create an instance of KernelSAX."""
    return KernelSAX()


def test_fit_with_lloyd(sample_data, ksax_instance):
    """Test the fit method of KernelSAX with LloydMaxQuantizer."""
    alphabets = ksax_instance.fit(sample_data)
    assert ksax_instance.is_fitted is True
    assert len(alphabets) == len(sample_data) // 2
    assert all(isinstance(alphabet, str) for alphabet in alphabets)


def test_fit_with_quantile(sample_data):
    ksax_instance = KernelSAX(boundary_estimator="quantile")
    alphabets = ksax_instance.fit(sample_data, paa_window_size=2)
    assert ksax_instance.is_fitted is True
    assert len(alphabets) == len(sample_data) // 2
    assert all(isinstance(alphabet, str) for alphabet in alphabets)


def test_encode_without_fit(ksax_instance):
    with pytest.raises(AssertionError) as excinfo:
        ksax_instance.encode_with_lloyd_boundaries()
    assert "fit() method must be called before encode()." in str(excinfo.value)


def test_decode_without_fit(ksax_instance):
    with pytest.raises(AssertionError) as excinfo:
        ksax_instance.decode_with_lloyd_boundaries()
    assert "fit() method must be called before decode()." in str(excinfo.value)


def test_encode_decode(ksax_instance, sample_data):
    ksax_instance.fit(sample_data, paa_window_size=2)
    encoded = ksax_instance.encode_with_lloyd_boundaries()
    decoded = ksax_instance.decode_with_lloyd_boundaries()

    assert len(encoded) == len(decoded)
    for decoded_range in decoded:
        assert (
            len(decoded_range) == 2
        )  # Each decoded entry should be a tuple of 2 values.
        assert (
            decoded_range[0] <= decoded_range[1]
        )  # The first value should be less than or equal to the second.
