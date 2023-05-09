import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class PAA:
    """
    Class implementing the Piecewise Aggregate Approximation algorithm for univariate time series.
    """

    def __init__(self, num_segments: int):
        """
        Parameters:
        -----------
        num_segments: int
            The number of segments to use in the PAA representation of the time series.
        """
        self.num_segments = num_segments

    def fit_transform(self, ts: np.ndarray) -> np.ndarray:
        """
        Computes the PAA representation of a given time series.

        Parameters:
        -----------
        ts: np.ndarray
            The time series to be transformed.

        Returns:
        --------
        np.ndarray
            The PAA representation of the time series.
        """
        if not isinstance(ts, np.ndarray):
            raise TypeError("Time series must be a numpy array.")
        if ts.ndim != 1:
            raise ValueError("Time series must be univariate.")
        if len(ts) < self.num_segments:
            raise ValueError(
                "Number of segments must be smaller than the time series length."
            )

        n = len(ts)
        seg_size = n // self.num_segments
        remainder = n % self.num_segments

        if remainder == 0:
            ts_paa = np.mean(ts.reshape(self.num_segments, seg_size), axis=1)
        else:
            ts_paa = np.concatenate(
                (
                    np.mean(ts[:remainder].reshape(seg_size + 1, -1), axis=1),
                    np.mean(ts[remainder:].reshape(seg_size, -1), axis=1),
                )
            )

        # Repeat the PAA values for each segment to obtain the PAA representation of the whole time series
        ts_paa = np.repeat(ts_paa, seg_size)
        ts_paa = np.concatenate((ts_paa, np.repeat(ts_paa[-1], remainder)))

        return ts_paa


def generate_random_ts(n: int) -> np.ndarray:
    """
    Generates a random univariate time series of given length.

    Parameters:
    -----------
    n: int
        The length of the time series to be generated.

    Returns:
    --------
    np.ndarray
        The generated time series.

    """
    return np.random.randn(n)


def plot_paa(ts: np.ndarray, ts_paa: np.ndarray, num_segments: int):
    """
    Plots a time series and its PAA representation on the same plot.

    Parameters:
    -----------
    ts: np.ndarray
        The original time series to be plotted.
    ts_paa: np.ndarray
        The PAA representation of the time series to be plotted.
    num_segments: int
        The number of segments used to compute the PAA representation.

    """
    plt.plot(ts, label="Original")
    plt.plot(ts_paa, label=f"PAA ({num_segments} segments)")
    plt.legend()
    plt.show()
