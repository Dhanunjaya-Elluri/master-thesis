import numpy as np
from scipy import integrate
from pa_approx import PAA
from lloyd_max import LloydMaxQuantizer
from sklearn.neighbors import KernelDensity
from typing import Callable, List


class KernelSAX:
    def __init__(
        self,
        x: np.ndarray,
        alphabets: np.ndarray,
        kernel: str = "gaussian",
        bandwidth: float = 0.2,
        epochs: int = 100,
        verbose: bool = False,
        random_state: int = 42,
    ) -> None:
        assert x.ndim == 1, "Only univariate time series data is supported."
        assert alphabets.ndim > 0, "Number of alphabets must be greater than zero."
        assert kernel in [
            "gaussian",
            "epanechnikov",
        ], "Invalid kernel type. Supported kernels: 'gaussian', 'epanechnikov'."

        self.x = x
        self.alphabets = alphabets
        self.n_codewords = len(alphabets)
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.epochs = epochs
        self.verbose = verbose
        self.random_state = random_state
        self.is_fitted = False

    def estimate_density(self, paa_segments: np.ndarray) -> Callable[[float], float]:
        kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        kde.fit(paa_segments.reshape(-1, 1))
        density = lambda t: np.exp(kde.score_samples([[t]]))
        return density

    def find_cut_points(self, paa_window_size: int) -> np.ndarray:
        assert paa_window_size > 0, "PAA window size must be greater than zero."

        paa = PAA(window_size=paa_window_size)
        paa.fit(self.x)
        paa_segments = paa.transform(self.x)

        density = self.estimate_density(paa_segments)
        quantizer = LloydMaxQuantizer(
            paa_segments,
            density,
            self.n_codewords,
            self.epochs,
            self.verbose,
            self.random_state,
            "kmeans++",
        )
        self.boundaries, self.codewords = quantizer.fit()
        self.is_fitted = True
        return self.boundaries

    def encode(self, values: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "The find_cut_points() method has not been called yet. Please call find_cut_points before encode."
            )
        assert values.ndim == 1, "Only univariate time series data is supported."
        assert (
            len(self.boundaries) > 0
        ), "No cut points (boundaries) available. Call find_cut_points() first."

        codes = np.digitize(values.reshape(-1, 1), self.boundaries[1:], right=True)
        _codes = codes.copy()
        for idx, token in enumerate(self.alphabets):
            _codes[np.where(codes == idx)] = token
        return _codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "The find_cut_points() method has not been called yet. Please call find_cut_points before decode."
            )
        assert codes.ndim == 1, "Only univariate time series data is supported."

        codes = codes.reshape(-1, 1)

        if isinstance(codes, str):
            codes = list(codes)

        codes = np.array(codes)
        unique_codes = np.unique(codes)

        decoded_values = np.zeros_like(codes, dtype=float)
        for code in unique_codes:
            code_idx = np.argmax(self.alphabets == code)
            decoded_values[np.where(codes == code)] = self.codewords[code_idx]

        return decoded_values
