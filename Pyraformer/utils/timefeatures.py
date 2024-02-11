#!/usr/bin/env python
# coding: utf-8

"""Time Features section for Pyraformer model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """
    Base class for time feature extraction.

    This class is designed to be subclassed by specific time feature extractors.
    """

    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extracts the time feature from a given Pandas DatetimeIndex.

        Args:
            index (pd.DatetimeIndex): The datetime index to extract features from.

        Returns:
            np.ndarray: An array of the extracted time features.
        """
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """
    Extracts the 'second of minute' feature, encoded as values between [-0.5, 0.5].
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """
    Extracts the 'minute of hour' feature, encoded as values between [-0.5, 0.5].
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """
    Extracts the 'hour of day' feature, encoded as values between [-0.5, 0.5].
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """
    Extracts the 'day of week' feature, encoded as values between [-0.5, 0.5].
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """
    Extracts the 'day of month' feature, encoded as values between [-0.5, 0.5].
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """
    Extracts the 'day of year' feature, encoded as values between [-0.5, 0.5].
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """
    Extracts the 'month of year' feature, encoded as values between [-0.5, 0.5].
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """
    Extracts the 'week of year' feature, encoded as values between [-0.5, 0.5].
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features appropriate for the given frequency string.

    Args:
        freq_str (str): Frequency string (e.g., "12H", "5min", "1D").

    Returns:
        List[TimeFeature]: A list of time feature extractor instances.
    """
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
        Unsupported frequency {freq_str}
        The following frequencies are supported:
            Y   - yearly
                alias: A
            M   - monthly
            W   - weekly
            D   - daily
            B   - business days
            H   - hourly
            T   - minutely
                alias: min
            S   - secondly
        """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, timeenc: int, freq: str) -> np.ndarray:
    """
    Extracts time features from dates according to the specified encoding and frequency.

    Args:
        dates (pd.DataFrame or pd.Series): DataFrame or Series containing datetime information.
        timeenc (int): Encoding type (0 for categorical, 1 for continuous).
        freq (str): Frequency of the time series data (e.g., 'h' for hourly).

    Returns:
        np.ndarray: Array containing the extracted time features.
    """
    if timeenc == 0:
        dates["month"] = dates.date.apply(lambda row: row.month, 1)
        dates["day"] = dates.date.apply(lambda row: row.day, 1)
        dates["weekday"] = dates.date.apply(lambda row: row.weekday(), 1)
        dates["hour"] = dates.date.apply(lambda row: row.hour, 1)
        dates["minute"] = dates.date.apply(lambda row: row.minute, 1)
        dates["minute"] = dates.minute.map(lambda x: x // 15)
        freq_map = {
            "y": [],
            "m": ["month"],
            "w": ["month"],
            "d": ["month", "day", "weekday"],
            "b": ["month", "day", "weekday"],
            "h": ["month", "day", "weekday", "hour"],
            "t": ["month", "day", "weekday", "hour", "minute"],
        }
        return dates[freq_map[freq.lower()]].values
    if timeenc == 1:
        dates = pd.to_datetime(dates.date.values)
        return np.vstack(
            [feat(dates) for feat in time_features_from_frequency_str(freq)]
        ).transpose(1, 0)
