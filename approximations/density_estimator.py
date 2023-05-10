import numpy as np
import matplotlib.pyplot as plt


class TimeSeriesDensityEstimator:
    def __init__(self, bw: float):
        self.bw = bw
        self.time_series = np.array([])
        self.n = 0
        self.kde_values = np.array([])

    def epanechnikov_kernel(self, x: np.ndarray) -> np.ndarray:
        # Use numpy arrays for broadcasting
        x = np.asarray(x)
        kernel_values = np.zeros_like(x)

        # Compute kernel values for elements where |x| <= 1
        mask = abs(x) <= 1
        kernel_values[mask] = 0.75 * (1 - x[mask] ** 2)

        return kernel_values

    def evaluate_kernel_density(self, x: float) -> float:
        numerator = 0
        denominator = 0
        for xi in self.time_series:
            k = self.epanechnikov_kernel((x - xi) / self.bw)
            numerator += k
            denominator += 1
        return numerator / (denominator * self.bw)

    def fit(self, time_series: np.ndarray) -> None:
        self.time_series = time_series
        self.n = len(time_series)
        self.kde_values = np.zeros(self.n)
        for i, x in enumerate(self.time_series):
            self.kde_values[i] = self.evaluate_kernel_density(x)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        pdf = np.zeros(len(x))
        for i, xi in enumerate(x):
            pdf[i] = np.sum(
                self.epanechnikov_kernel((xi - self.time_series) / self.bw)
            ) / (self.n * self.bw)
        return pdf
