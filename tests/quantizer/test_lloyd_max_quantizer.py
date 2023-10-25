#!/usr/bin/env python
# coding: utf-8

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import pytest
import numpy as np
from scipy.stats import norm
from tqts.quantizer.lloyd_max import LloydMaxQuantizer


@pytest.fixture
def sample_data():
    # Create sample data for testing
    np.random.seed(42)
    return np.random.randn(1000)


def test_initialize_codewords_random():
    # Test random initialization of codewords
    quantizer = LloydMaxQuantizer([], lambda x: 0, 5, init_codewords="random")
    codewords = quantizer._initialize_codewords_random(0, 1, np.random.RandomState(42))
    assert len(codewords) == 5
    assert all(0 <= codeword <= 1 for codeword in codewords)


def test_fit(sample_data):
    # Test the fit method of LloydMaxQuantizer
    quantizer = LloydMaxQuantizer(sample_data, norm.pdf, 5)
    boundaries, codewords = quantizer.fit()

    # Check if the number of boundaries and codewords match the expected number
    assert len(boundaries) == 6  # n_codewords + 1
    assert len(codewords) == 5

    # Ensure that the boundaries are sorted in ascending order
    assert np.all(np.diff(boundaries) >= 0)

    # Ensure that the codewords are within the range of the input data
    assert all(
        min(sample_data) <= codeword <= max(sample_data) for codeword in codewords
    )
