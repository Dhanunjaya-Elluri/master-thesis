#!/usr/bin/env python
# coding: utf-8

"""Utility for Dataset modules."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

from typing import List
from datetime import datetime, timedelta
import pandas as pd


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


def text_to_df(file_path: str, start_datetime: str, interval: int = 4) -> pd.DataFrame:
    """Processes a text file containing a sequence of characters into a pandas DataFrame.

    Args:
        file_path (str): Path to the text file.
        start_datetime (str): Start date and time of the dataset.
        interval (int, optional): Interval between timestamps. Defaults to 4.

    Returns:
        pd.DataFrame: Pandas dataframe of the dataset.
    """
    with open(file_path, "r") as file:
        data = file.read()

    # Character to index mapping
    char_to_idx = {char: idx for idx, char in enumerate(sorted(set(data)))}

    timestamps = generate_timestamps(start_datetime, len(data), interval)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sequence": list(data),
            "encoded_sequence": [char_to_idx[char] for char in data],
        }
    )
    return df
