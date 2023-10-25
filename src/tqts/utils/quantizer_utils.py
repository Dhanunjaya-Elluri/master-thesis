#!/usr/bin/env python
# coding: utf-8

"""General utility functions."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

from typing import Tuple

import numpy as np


def character_distance(char1: str, char2: str) -> int:
    """
    Calculates the distance between two characters.

    Args:
        char1 (str): The first character.
        char2 (str): The second character.

    Returns:
        int: The distance between the two characters.
    """
    return abs(ord(char1) - ord(char2))


def character_distance_between_strings(s1: str, s2: str) -> int:
    """
    Calculates the character-wise distance between two strings of the same length.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        int: The character-wise distance between the two strings.

    Raises:
        ValueError: If the input strings have different lengths.
    """
    if len(s1) != len(s2):
        raise ValueError("Strings must be of the same length")

    total_distance = 0

    for char1, char2 in zip(s1, s2):
        total_distance += character_distance(char1, char2)

    return total_distance


def calculate_quantile_levels(
    data: np.ndarray, num_quantiles: int
) -> Tuple[list, np.ndarray]:
    """
    Calculate the quantile levels for the given data.

    Args:
        data (np.ndarray): The input data.
        num_quantiles (int): The number of quantiles to calculate.

    Returns:
        Tuple[list, np.ndarray]: Tuple containing the quantile levels and quantiles.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # Base level for each quantile
    base = 100.0

    quantile_levels = [0]
    for i in range(1, num_quantiles - 1):
        level = (base * i) / (num_quantiles - 1)
        quantile_levels.append(level)

    quantile_levels.append(100)
    quantiles = np.percentile(data, quantile_levels)

    return quantile_levels, quantiles


def find_symbol(paa_value: float, quantiles: np.ndarray, ascii_codes: list) -> str:
    """
    Find the symbol for the given PAA value.

    Args:
        paa_value (float): The PAA value.
        quantiles (np.ndarray): The quantiles.
        ascii_codes (list): The ASCII codes for the symbols.

    Returns:
        str: The symbol for the given PAA value.
    """
    for i in range(1, len(quantiles)):
        if quantiles[i - 1] <= paa_value <= quantiles[i]:
            return ascii_codes[i - 1]
    return ascii_codes[-1]  # For the last segment, which is <= max value
