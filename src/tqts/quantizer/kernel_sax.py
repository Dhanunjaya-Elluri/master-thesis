#!/usr/bin/env python
# coding: utf-8

"""Kernel SAX (K-SAX) is a kernel-based symbolic aggregate approximation technique for time series data."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

from typing import Tuple
import numpy as np
from scipy.interpolate import interp1d

from tqts.quantizer.paa import PAA
from tqts.quantizer.lloyd_max import LloydMaxQuantizer
from sklearn.neighbors import KernelDensity

from tqts.utils.quantizer_utils import calculate_quantile_levels, find_symbol


class KernelSAX:
    def __init__(
        self,
        kernel: str = "gaussian",
        n_alphabet: int = 7,
        bandwidth: float = 3,
        boundary_estimator: str = "lloyd-max",
        epochs: int = 100,
        random_state: int = 42,
    ) -> None:
        self.verbose = None
        self._validate_parameters(kernel, boundary_estimator)
        self.n_alphabet = n_alphabet
        self.kernel = kernel
        self.boundary_estimator = boundary_estimator
        self.bandwidth = bandwidth
        self.epochs = epochs
        self.random_state = random_state
        self.is_fitted = False
        self.paa_series = None
        self._initialize_attributes()
        self.ascii_codes = [
            chr(i) for i in range(65, 65 + self.n_alphabet)
        ]  # ASCII codes for uppercase letters

    @staticmethod
    def _validate_parameters(kernel, boundary_estimator):
        assert boundary_estimator in [
            "lloyd-max",
            "quantile",
        ], "Invalid boundary estimator. Supported estimators: 'lloyd-max', 'quantile'."
        assert kernel in [
            "gaussian",
            "epanechnikov",
        ], "Invalid kernel type. Supported kernels: 'gaussian', 'epanechnikov'."

    def _initialize_attributes(self):
        self.assigned_codewords = None
        self.codewords = None
        self.boundaries = None
        self.alphabets = None
        self.codeword_to_alphabet = None

    def _estimate_density(
        self, paa_series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate pdf of timeseries using kernel density estimation.

        Args:
            paa_series (np.ndarray): PAA segments of the input time series.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the x and y values of the estimated density.
        """
        kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        kde.fit(paa_series.reshape(-1, 1))

        # Range of values within the min and max of the PAA segments, for density estimation
        x_d = np.linspace(np.min(paa_series), np.max(paa_series), 1000)[:, np.newaxis]

        log_density = kde.score_samples(x_d)
        density = np.exp(log_density)
        return x_d, density

    def _calculate_lloyd_max_boundaries(
        self, x_d: np.ndarray, density: np.ndarray
    ) -> None:
        """Calculate the boundaries for the Lloyd-Max quantizer.

        Args:
            x_d (np.ndarray): X values for the estimated density.
            density (np.ndarray): Y values for the estimated density.

        Returns:
            None
        """
        # Find the quantiles of the estimated density
        x_d_flatten = x_d.flatten()
        density_interp = interp1d(x_d_flatten, density)
        quantizer = LloydMaxQuantizer(
            x=x_d_flatten,
            density_func=density_interp,
            n_codewords=self.n_alphabet,
            epochs=self.epochs,
            verbose=self.verbose,
            random_state=self.random_state,
            init_codewords="random",
        )
        self.boundaries, self.codewords = quantizer.fit()
        self.is_fitted = True
        self.alphabets = self.encode_with_lloyd_boundaries()

    def _calculate_quantile_boundaries(self, x: np.ndarray) -> None:
        """Calculate the boundaries for the quantile-based quantizer.

        Args:
            x (np.ndarray): The input time series.

        Returns:
            None
        """
        quantile_levels, quantiles = calculate_quantile_levels(
            data=x, num_quantiles=self.n_alphabet
        )
        self.alphabets = [
            find_symbol(paa_value, quantiles, self.ascii_codes)
            for paa_value in self.paa_series
        ]
        self.is_fitted = True

    def fit(self, x: np.ndarray, paa_window_size: int, verbose: bool = True) -> list:
        assert paa_window_size > 0, "PAA window size must be greater than zero."
        self.verbose = verbose

        paa = PAA(window_size=paa_window_size)
        paa.fit(x)
        self.paa_series = paa.transform(x)

        x_d, density = self._estimate_density(self.paa_series)

        if self.boundary_estimator == "lloyd-max":
            self._calculate_lloyd_max_boundaries(x_d, density)

        elif self.boundary_estimator == "quantile":
            self._calculate_quantile_boundaries(x)

        assert len(self.paa_series) == len(
            self.alphabets
        ), "Length of PAA series and alphabets must be equal."
        return self.alphabets

    def encode_with_lloyd_boundaries(self) -> list:
        """Assign codewords to PAA segments.

        Returns:
            list: List of codewords assigned to each PAA segment.
        """
        assert self.is_fitted, "fit() method must be called before encode()."
        assignments = []
        for value in self.paa_series:
            assigned = False
            # Find the region that this PAA value belongs to
            for i in range(1, len(self.boundaries)):
                if self.boundaries[i - 1] <= value < self.boundaries[i]:
                    assignments.append(self.codewords[i - 1])
                    assigned = True
                    break
            if not assigned:
                assignments.append(self.codewords[-1])

        unique_codewords = np.unique(assignments)
        self.codeword_to_alphabet = dict(zip(unique_codewords, self.ascii_codes))
        return [self.codeword_to_alphabet[codeword] for codeword in assignments]

    def decode_with_lloyd_boundaries(self) -> list:
        """
        Convert alphabets back to their corresponding value ranges in the original time series.

        Returns:
            list: List of values corresponding to each alphabet.
        """
        assert self.is_fitted, "fit() method must be called before decode()."
        alphabet_to_codeword = {
            alphabet: codeword
            for codeword, alphabet in self.codeword_to_alphabet.items()
        }
        # Map the symbols back to their codewords
        codewords = [alphabet_to_codeword[alphabet] for alphabet in self.alphabets]
        # Initialize the list that will hold the value ranges
        original_values = []

        # For each codeword, find the corresponding range in the original time series
        for codeword in codewords:
            for i in range(len(self.boundaries) - 1):
                if self.boundaries[i] <= codeword < self.boundaries[i + 1]:
                    # Append the range that the codeword represents
                    original_values.append((self.boundaries[i], self.boundaries[i + 1]))
                    break

        return original_values

    def plot_with_boundaries(self) -> None:
        """TODO: Add docstring."""
        pass

    def plot_with_quantiles(self) -> None:
        """TODO: Add docstring."""
        pass
