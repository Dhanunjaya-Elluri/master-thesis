#!/usr/bin/env python
# coding: utf-8

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import numpy as np
import pytest

from tqts.utils.quantizer_utils import (
    calculate_quantile_levels,
    character_distance,
    character_distance_between_strings,
    find_symbol,
)


def test_character_distance():
    # Test with two equal characters
    assert character_distance("a", "a") == 0

    # Test with two unequal characters
    assert character_distance("a", "c") == 2


def test_character_distance_between_strings():
    # Test equal strings with character 'a' and 'c' at the same positions
    assert character_distance_between_strings("abc", "cbc") == 2

    # Test equal strings with character 'a' and 'c' at different positions
    assert character_distance_between_strings("abc", "cba") == 4

    # Test unequal strings
    with pytest.raises(ValueError):
        character_distance_between_strings("abc", "abcd")


def test_calculate_quantile_levels():
    # Test with a numpy array
    import numpy as np

    data = np.random.randn(1000)
    quantile_levels, quantiles = calculate_quantile_levels(data, 5)
    assert len(quantile_levels) == 5
    assert len(quantiles) == 5

    # Test with a list
    data = list(np.random.randn(1000))
    quantile_levels, quantiles = calculate_quantile_levels(data, 5)
    assert len(quantile_levels) == 5
    assert len(quantiles) == 5

    # Test with a pandas Series
    import pandas as pd

    data = pd.Series(np.random.randn(1000))
    quantile_levels, quantiles = calculate_quantile_levels(data, 5)
    assert len(quantile_levels) == 5
    assert len(quantiles) == 5

    # Test with a pandas DataFrame
    data = pd.DataFrame(np.random.randn(1000))
    quantile_levels, quantiles = calculate_quantile_levels(data, 5)
    assert len(quantile_levels) == 5
    assert len(quantiles) == 5

    # Test with an invalid data type
    with pytest.raises(TypeError):
        calculate_quantile_levels("abc", 5)


def test_find_symbol():
    ascii_codes = [chr(i) for i in range(97, 97 + 3)]  # a, b, c
    quantile_levels = np.array([0.0, 0.5, 1.0])
    paa_value_within = 0.25
    paa_value_last_segment = 0.75

    # Test for the within segment condition
    symbol = find_symbol(paa_value_within, quantile_levels, ascii_codes)
    assert symbol == "a", f"Error: expected 'a', but got {symbol}"

    # Test for the last segment condition
    symbol = find_symbol(paa_value_last_segment, quantile_levels, ascii_codes)
    assert symbol == "b", f"Error: expected 'b', but got {symbol}"
