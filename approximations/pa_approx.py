import numpy as np
import matplotlib.pyplot as plt
from typing import List


class PAA:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.n_windows = None

    def fit(self, X: np.ndarray) -> None:
        """
        Compute the number of windows for the input time series.
        :param X: np.ndarray, shape (n_samples,)
            The input time series.
        """
        n_samples = len(X)
        self.n_windows = n_samples // self.window_size

        if self.n_windows == 0:
            raise ValueError("Window size is too large for the input time series.")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input time series using Piecewise Aggregate Approximation (PAA).
        :param X: np.ndarray, shape (n_samples,)
            The input time series.
        :return: np.ndarray, shape (n_windows,)
            The PAA segments obtained by averaging each window.
        """
        if self.n_windows is None:
            raise ValueError(
                "The fit method has not been called yet. Please call fit before transform."
            )

        # reshape X into windows
        X_windows = X[: self.n_windows * self.window_size].reshape(
            self.n_windows, self.window_size
        )

        # compute mean of each window
        paa_segments = np.mean(X_windows, axis=1)

        return paa_segments

    def plot(self, X: np.ndarray) -> None:
        """
        Plot the PAA segments obtained from an input time series.
        :param X: np.ndarray, shape (n_samples,)
            The input time series.
        """
        self.fit(X)
        paa_segments = self.transform(X)

        # plot original time series
        plt.plot(X)

        # plot vertical lines with a gap of window size
        for i in range(1, self.n_windows):
            plt.axvline(i * self.window_size - 0.5, color="gray", linestyle="--")

        # plot PAA segments
        plt.plot(np.repeat(paa_segments, self.window_size))

        plt.show()
