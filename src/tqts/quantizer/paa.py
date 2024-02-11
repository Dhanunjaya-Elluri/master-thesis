#!/usr/bin/env python
# coding: utf-8

"""Piecewise Aggregate Approximation (PAA) module."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

import matplotlib.pyplot as plt
import numpy as np


class PAA:
    def __init__(self, window_size: int = None, n_windows: int = None) -> None:
        # Only one of window_size or n_windows should be provided
        if window_size is not None and n_windows is not None:
            raise ValueError("Provide either 'window_size' or 'n_windows', not both.")
        elif window_size is None and n_windows is None:
            raise ValueError("Either 'window_size' or 'n_windows' must be provided.")

        self.window_size = window_size
        self.n_windows = n_windows

    def fit(self, x: np.ndarray) -> None:
        """
        Compute the number of windows or window size for the input time series, depending on the input given.
        Args:
            x (np.ndarray): The input time series.
        """
        n_samples = len(x)

        if self.window_size:
            self.n_windows = n_samples // self.window_size
            if self.n_windows == 0:
                raise ValueError("Window size is too large for the input time series.")
        else:
            self.window_size = n_samples // self.n_windows
            if self.window_size == 0:
                raise ValueError("Too many windows for the input time series.")

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform the input time series using Piecewise Aggregate Approximation (PAA).
        Args:
            x (np.ndarray): The input time series.
        Returns:
            np.ndarray: The PAA segments obtained by averaging each window.
        """
        if self.n_windows is None or self.window_size is None:
            raise ValueError(
                "The fit method has not been called yet. Please call fit before transform."
            )

        # reshape X into windows
        x = np.array(x)
        x_windows = x[: self.n_windows * self.window_size].reshape(
            self.n_windows, self.window_size
        )

        # compute mean of each window
        paa_segments = np.mean(x_windows, axis=1)

        return paa_segments

    def inverse_transform(self, paa_segments: np.ndarray) -> np.ndarray:
        """
        Perform the inverse transform to reconstruct the time series from the PAA segments.
        Args:
            paa_segments (np.ndarray): The PAA segments.
        Returns:
            np.ndarray: The reconstructed time series.
        """
        if self.n_windows is None or self.window_size is None:
            raise ValueError(
                "The fit method has not been called yet. Please call fit before inverse_transform."
            )

        # If there are less segments than expected, fill in with the average of the existing segments
        if len(paa_segments) < self.n_windows:
            average_value = np.mean(paa_segments)
            paa_segments = np.append(
                paa_segments, [average_value] * (self.n_windows - len(paa_segments))
            )

        # Repeat each segment value window_size times to reconstruct the original time series length
        repeated_segments = np.repeat(paa_segments, self.window_size)

        return repeated_segments

    def plot(self, x: np.ndarray) -> None:
        """
        Plot the PAA segments obtained from an input time series.
        Args:
            x (np.ndarray): The input time series.
        """
        self.fit(x)
        paa_segments = self.transform(x)

        # plot original time series
        plt.plot(x)

        # plot vertical lines with a gap of window size
        for i in range(1, self.n_windows):
            plt.axvline(i * self.window_size - 0.5, color="gray", linestyle="--")

        # plot PAA segments
        plt.plot(np.repeat(paa_segments, self.window_size))

        plt.show()
