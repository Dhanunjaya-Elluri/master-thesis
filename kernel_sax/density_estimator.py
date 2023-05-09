import numpy as np
from scipy.stats import norm


class KDE:
    def __init__(self, data, num_bins=50):
        """
        Parameters
        data: 1D array
        num_bins: int
        """
        self.data = np.asarray(data)
        self.num_bins = num_bins
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.bin_edges = np.linspace(
            self.mean - 3 * self.std, self.mean + 3 * self.std, self.num_bins + 1
        )
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.histogram = np.zeros(self.num_bins)
        self.weights = None
        self.kernel = norm()
        self.distribution = None

    def _paa(self, data):
        window_size = int(len(data) / self.num_bins)
        data = data[: window_size * self.num_bins]
        data = data.reshape((self.num_bins, window_size))
        return np.mean(data, axis=1)

    def fit(self):
        paa_data = self._paa(self.data)
        for i in range(len(paa_data)):
            bin_idx = np.argmin(np.abs(self.bin_centers - paa_data[i]))
            self.histogram[bin_idx] += 1
        self.weights = self.histogram / (
            np.sum(self.histogram) * (self.bin_edges[1] - self.bin_edges[0])
        )
        self.distribution = sum(
            [
                self.weights[i]
                * self.kernel.pdf(
                    self.bin_centers, loc=self.bin_centers[i], scale=self.std
                )
                for i in range(self.num_bins)
            ]
        )

    def pdf(self, x):
        return np.interp(x, self.bin_centers, self.distribution)
