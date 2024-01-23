#!/usr/bin/env python
# coding: utf-8

"""Utility for Time Features module."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"


from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeatures:
    """
    Abstract base class for time feature extraction.

    Child classes must implement the __call__ method to provide feature extraction.
    """

    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract the time feature from a given DatetimeIndex."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class SecondOfMinute(TimeFeatures):
    """Second of the minute encoded between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59 - 0.5


class MinuteOfHour(TimeFeatures):
    """Minute of the hour encoded between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59 - 0.5


class HourOfDay(TimeFeatures):
    """Hour of the day encoded between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23 - 0.5


class DayOfWeek(TimeFeatures):
    """Day of the week encoded between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6 - 0.5


class DayOfMonth(TimeFeatures):
    """Day of the month encoded between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30 - 0.5


class DayOfYear(TimeFeatures):
    """Day of the year encoded between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365 - 0.5


class WeekOfYear(TimeFeatures):
    """Week of the year encoded between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52 - 0.5


class MonthOfYear(TimeFeatures):
    """Month of the year encoded between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11 - 0.5


def time_features_from_freq(freq: str) -> List[TimeFeatures]:
    """
    Returns a list of time feature extractor instances for a given frequency.

    Args:
        freq (str): Frequency string as per pandas frequency notation.

    Returns:
        List[TimeFeature]: List of time feature extractor instances.

    Raises:
        ValueError: If the frequency is not supported.
    """
    feature_by_offset = {
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.YearEnd: [],
    }
    offset = to_offset(freq)
    for offset_type, features_classes in feature_by_offset.items():
        if isinstance(offset, offset_type):
            return [feat() for feat in features_classes]
    supported_freq_msg = f"""
        Supported frequencies are:
            S - second
            T - minute
            H - hour
            D - day
            B - business day
            W - week
            M - month
            Q - quarter
            A - year
        """
    raise ValueError(f"Unsupported frequency {freq}. {supported_freq_msg}")


def time_features(
    dates: pd.DataFrame, freq: str = "4h", time_enc: int = 1
) -> np.ndarray:
    """
    Returns a numpy array of time features for given dates and frequency.

    Args:
        dates (pd.DatetimeIndex): DatetimeIndex to extract features from.
        freq (str): Frequency of the dataset. Defaults to "4h".
        time_enc (int): Time encoding to use. Defaults to 1.

    Returns:
        np.ndarray: Array of time features.
    """
    if time_enc == 0:
        dates["month"] = dates.date.apply(lambda row: row.month, 1)
        dates["day"] = dates.date.apply(lambda row: row.day, 1)
        dates["weekday"] = dates.date.apply(lambda row: row.weekday(), 1)
        dates["hour"] = dates.date.apply(lambda row: row.hour, 1)
        dates["minute"] = dates.date.apply(lambda row: row.minute, 1)
        dates["minute"] = dates.minute.map(lambda x: x // 30)
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
    if time_enc == 1:
        dates = pd.to_datetime(dates.timestamp.values)
        return np.vstack(
            [feat(dates) for feat in time_features_from_freq(freq)]
        ).transpose(1, 0)
