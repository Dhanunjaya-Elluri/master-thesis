#!/usr/bin/env python
# coding: utf-8

"""Utility for Dataset modules."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

from datetime import datetime, timedelta
from typing import List

import numpy as np


def generate_timestamps(start_datetime: str, length: int, interval: int) -> List[str]:
    """Generates a list of timestamps.

    Args:
        start_datetime (str): Start date and time of the dataset.
        length (int): Length of the dataset.
        interval (int, optional): Interval between timestamps.

    Returns:
        List[str]: List of timestamps.
    """
    start_datetime = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
    time_stamps = [
        start_datetime + timedelta(hours=i * interval) for i in range(length)
    ]
    return [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in time_stamps]


def vectorized_find_character(values, lower_boundaries, upper_boundaries, alphabets):
    # Initialize an array to store the characters
    characters = np.full(values.shape, None)

    # Iterate over each boundary and assign characters
    for i, (lower, upper) in enumerate(zip(lower_boundaries, upper_boundaries)):
        mask = (values >= lower) & (values < upper)
        characters[mask] = alphabets[i]

    return characters
