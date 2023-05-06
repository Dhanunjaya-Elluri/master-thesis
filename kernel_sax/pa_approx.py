import numpy as np


class PAA:
    """
    Piecewise Aggregate Approximation (PAA) for time series data.
    """

    def __init__(self, n_pieces: int) -> None:
        """
        Parameters:
        -----------
        n_pieces: int
            Number of pieces to aggregate the time series into.
        """
        self.n_pieces = n_pieces

    def fit_transform(self, ts: np.ndarray) -> np.ndarray:
        """
        Transform a time series into its PAA representation.

        Parameters:
        -----------
        ts: ndarray of shape (n_samples,)
            Time series to transform.

        Returns:
        --------
        paa: ndarray of shape (n_pieces,)
            PAA representation of the input time series.
        """
        n_samples, n_features = ts.shape
        window_size = n_samples // self.n_pieces
        ts_paa = np.zeros((n_samples, self.n_pieces))

        for i in range(self.n_pieces):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            if i == self.n_pieces - 1:
                end_idx = n_features
            ts_paa[:, i] = np.mean(ts[:, start_idx:end_idx], axis=1)

        return ts_paa
