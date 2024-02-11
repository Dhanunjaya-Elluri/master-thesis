#!/usr/bin/env python
# coding: utf-8

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"


import numpy as np
import pytest

from tqts.quantizer.lloyd_max import LloydMaxQuantizer


# Fixtures are functions that create data or set conditions for tests
@pytest.fixture
def sample_data():
    # Creating a sample density function for testing
    def density_func(x):
        return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

    return density_func, np.random.randn(100)


def test_initialization():
    """
    Test successful instantiation of the LloydMaxQuantizer class.
    """
    density_func = lambda x: np.exp(-(x**2))
    quantizer = LloydMaxQuantizer(density_func, n_codewords=2, random_state=42)

    assert quantizer is not None
    assert quantizer.n_codewords == 2
    assert quantizer.random_state == 42


def test_invalid_initialization():
    """
    Test LloydMaxQuantizer instantiation with invalid initialization method.
    """
    density_func = lambda x: np.exp(-(x**2))

    with pytest.raises(AssertionError):
        LloydMaxQuantizer(
            density_func, n_codewords=2, init_codewords="unsupported_method"
        )


@pytest.mark.usefixtures("sample_data")
def test_fit_method(sample_data):
    """
    Test the fit method with sample data.
    """
    density_func, data = sample_data
    quantizer = LloydMaxQuantizer(density_func, n_codewords=2, random_state=42)

    boundaries, codewords = quantizer.fit(data)

    assert boundaries is not None
    assert codewords is not None
    assert len(boundaries) == 3  # n_codewords + 1
    assert len(codewords) == 2  # n_codewords


@pytest.mark.usefixtures("sample_data")
def test_convergence(sample_data):
    """
    Test whether the algorithm converges within a maximum number of epochs.
    """
    density_func, data = sample_data
    quantizer = LloydMaxQuantizer(
        density_func, n_codewords=2, random_state=42, verbose=False
    )

    boundaries, codewords = quantizer.fit(
        data, epochs=1000
    )  # assuming convergence within 1000 epochs

    # Check that the method does return arrays and they have the correct shape
    assert isinstance(boundaries, np.ndarray), "Boundaries is not an array"
    assert isinstance(codewords, np.ndarray), "Codewords is not an array"
    assert len(boundaries) == 3, "Unexpected number of boundaries"
    assert len(codewords) == 2, "Unexpected number of codewords"

    # Check that the boundaries are sorted (each next boundary should be greater than the previous)
    assert np.all(
        np.diff(boundaries) > 0
    ), "Boundaries are not sorted in ascending order"

    # Check that codewords are within the range of boundaries
    for i, codeword in enumerate(codewords):
        assert (
            boundaries[i] <= codeword <= boundaries[i + 1]
        ), f"Codeword {i} is out of bounds"


def test_integration_failure():
    """
    Test the behavior of the algorithm if the integration fails.
    """

    # Using a faulty density function to induce integration failure
    def faulty_density_func(x):
        raise ValueError("Integration failure induced for testing.")

    data = np.random.randn(100)
    quantizer = LloydMaxQuantizer(
        faulty_density_func, n_codewords=2, random_state=42, verbose=False
    )

    # Depending on the expected behavior, you can assert the exception or check the state of the quantizer
    with pytest.raises(ValueError):
        quantizer.fit(data)
