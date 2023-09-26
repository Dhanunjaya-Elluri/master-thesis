__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import pytest
import numpy as np
from scipy.stats import norm
from approximations.lloyd_max import LloydMaxQuantizer


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


@pytest.mark.skip(reason="Not implemented yet")
def test_initialize_codewords_kmeanspp():
    # Test k-means++ initialization of codewords
    quantizer = LloydMaxQuantizer(sample_data, lambda x: 0, 5, init_codewords="kmeans++")

    codewords = quantizer._initialize_codewords_kmeanspp(0, 1, np.random.RandomState(42))
    assert len(codewords) == 5
    assert all(0 <= codeword <= 1 for codeword in codewords)


def test_calculate_distances():
    # Test the calculation of distances between data points and codewords
    data = np.array([1.0, 2.0, 3.0])
    codewords = np.array([2.0, 4.0])
    distances = LloydMaxQuantizer._calculate_distances(data, codewords)
    expected_distances = np.array([4.0, 2.0, 2.0])
    assert np.allclose(distances, expected_distances)


def test_calculate_probabilities():
    # Test the calculation of probabilities based on distances
    distances = np.array([1.0, 2.0, 1.0])
    probabilities = LloydMaxQuantizer._calculate_probabilities(distances)
    expected_probabilities = np.array([1.0, 0.5, 1.0])
    assert np.allclose(probabilities, expected_probabilities)


def test_sample_new_codeword():
    # Test sampling a new codeword based on probabilities
    data = np.array([1.0, 2.0, 3.0])
    probabilities = np.array([0.4, 0.5, 0.1])
    rng = np.random.RandomState(42)
    sampled_codeword = LloydMaxQuantizer._sample_new_codeword(data, probabilities, rng)
    assert sampled_codeword in data


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
    assert all(min(sample_data) <= codeword <= max(sample_data) for codeword in codewords)


# Add more test cases as needed

if __name__ == "__main__":
    pytest.main()
