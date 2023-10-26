#!/usr/bin/env python
# coding: utf-8

"""Kernel SAX (K-SAX) is a kernel-based symbolic aggregate approximation technique for time series data."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter

from tqts.quantizer.paa import PAA
from tqts.quantizer.lloyd_max import LloydMaxQuantizer
from sklearn.neighbors import KernelDensity

from tqts.utils.quantizer_utils import calculate_quantile_levels, find_symbol

root_path = Path(__file__).parent.parent.parent


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
        self.x = None
        self.x_d_flatten = None
        self.density = None
        self.assigned_codewords = None
        self.codewords = None
        self.boundaries = None
        self.alphabets = None
        self.codeword_to_alphabet = None
        self.quantiles = None
        self.quantile_levels = None

    def _estimate_density(
            self, paa_series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate pdf of timeseries using kernel density estimation.

        Args:
            paa_series (np.ndarray): PAA segments of the input time series.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the x_d flatten and y values of the estimated density.
        """
        kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        kde.fit(paa_series.reshape(-1, 1))

        # Range of values within the min and max of the PAA segments, for density estimation
        x_d = np.linspace(np.min(paa_series), np.max(paa_series), 1000)[:, np.newaxis]

        log_density = kde.score_samples(x_d)
        density = np.exp(log_density)
        return x_d.flatten(), density

    def _calculate_lloyd_max_boundaries(
            self, x_d_flatten: np.ndarray, density: np.ndarray
    ) -> None:
        """Calculate the boundaries for the Lloyd-Max quantizer.

        Args:
            x_d_flatten (np.ndarray): Flattened X values for the estimated density.
            density (np.ndarray): Y values for the estimated density.

        Returns:
            None
        """
        # Find the quantiles of the estimated density
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
        self.quantile_levels, self.quantiles = calculate_quantile_levels(
            data=x, num_quantiles=self.n_alphabet
        )
        self.alphabets = [
            find_symbol(paa_value, self.quantiles, self.ascii_codes)
            for paa_value in self.paa_series
        ]
        self.is_fitted = True

    def fit(self, x: np.ndarray, paa_window_size: int, verbose: bool = True) -> list:
        assert paa_window_size > 0, "PAA window size must be greater than zero."
        self.verbose = verbose
        self.x = x

        paa = PAA(window_size=paa_window_size)
        paa.fit(self.x)
        self.paa_series = paa.transform(self.x)

        self.x_d_flatten, self.density = self._estimate_density(self.paa_series)

        if self.boundary_estimator == "lloyd-max":
            self._calculate_lloyd_max_boundaries(self.x_d_flatten, self.density)

        elif self.boundary_estimator == "quantile":
            self._calculate_quantile_boundaries(self.x)

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

    def save_alphabets(self, path: str) -> None:
        """Save the codewords to a file.

        Args:
            path (str): Path to save the codewords.

        Returns:
            None
        """
        assert self.is_fitted, "fit() method must be called before saving codewords."
        formatted_alphabets = " ".join(self.alphabets)
        with open(path, "w") as f:
            f.write(formatted_alphabets)
        print(f"Alphabets saved to {path}")

    def plot_with_boundaries(self, path: str) -> None:
        """Plot the PAA segments, assigned symbols, and density estimation.

        Args:
            path (str): Path to save the plot.

        Returns:
            None
        """
        assert self.is_fitted, "fit() method must be called before plotting."
        fig, ax1 = plt.subplots(figsize=(16, 6))
        ax1.plot(self.x, color="blue", label="Original Time Series", alpha=0.5)

        # Calculate the positions of the PAA segments to plot them correctly (middle of the segment)
        positions = np.linspace(0, len(self.x) - 1, len(self.paa_series) + 1)
        mid_positions = (positions[:-1] + positions[1:]) / 2
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.boundaries[1:-1])))

        # Plotting PAA segments with their symbols
        for idx, (pos, alphabet) in enumerate(zip(mid_positions, self.alphabets)):
            color_index = self.alphabets.index(alphabet)
            ax1.plot((pos, pos), (self.paa_series[idx], self.paa_series[idx]), color='black', linestyle='-',
                     linewidth=2)
            ax1.text(pos, self.paa_series[idx], self.alphabets[idx], fontsize=12, ha='center', va='center_baseline',
                     bbox=dict(boxstyle="round,pad=0.1", alpha=0.5, color=f"C{color_index}"))

        # Plotting the Lloyd-Max boundaries
        for boundary, color in zip(self.boundaries[1:-1],
                                   colors):  # excluding the first and last, as they are -inf and inf
            ax1.axhline(y=boundary, color=color, linestyle='--')

        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax1)

        # Append axes to the right of ax1, with 20% width of ax1
        ax2 = divider.append_axes("left", size="20%", pad=0.05)

        # Plotting the density estimation vertically
        ax2.fill_betweenx(self.x_d_flatten, 0, self.density, color='gray', alpha=0.5)
        ax2.set_xlabel('Density')

        # Invert the x-axis for the density plot to have it visually make sense with the plot on the right
        ax2.invert_xaxis()

        # Adding necessary plot details
        ax1.set_title('Time Series with lloyd-max boundaries')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')

        # Adding legends and labels
        ax1.legend(loc='upper left')

        # Show the plot
        plt.tight_layout()

        # Save the plot to root_path/images
        plt.savefig(path + "kernel_sax_with_boundaries.png", dpi=300)
        print(f"Plot saved to {path}")

    def plot_with_quantiles(self, path: str) -> None:
        """Plot the PAA segments, assigned symbols, and density estimation.

        Args:
            path (str): Path to save the plot.

        Returns:
            None
        """
        fig, ax1 = plt.subplots(figsize=(16, 6))

        ax1.plot(self.x, color='blue', label='Original Time Series', alpha=0.5)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.set_title('Time Series with Quantile Lines')

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.quantiles)))

        for i, (quantile, color) in enumerate(zip(self.quantiles, colors)):
            label = f"{self.quantile_levels[i]:.2f}th Percentile"  # Label for the legend
            ax1.axhline(y=quantile, color=color, linestyle='--', label=label)

        # Adding PAA segments and symbols
        # Calculate the positions of the PAA segments to plot them correctly (middle of the segment)
        positions = np.linspace(0, len(self.x) - 1, len(self.paa_series) + 1)
        mid_positions = (positions[:-1] + positions[1:]) / 2

        # Plotting PAA segments with their symbols
        for idx, (pos, symbol) in enumerate(zip(mid_positions, self.alphabets)):
            color_index = self.alphabets.index(symbol)
            ax1.plot((pos, pos), (self.paa_series[idx], self.paa_series[idx]), color='black', linestyle='-', linewidth=2)
            ax1.text(pos, self.paa_series[idx], self.alphabets[idx], fontsize=12, ha='center', va='center_baseline',
                     bbox=dict(boxstyle="round,pad=0.1", alpha=0.5, color=f'C{int(color_index)}'))
        # Creating the density plot again as per the previous steps
        divider = make_axes_locatable(ax1)
        ax2 = divider.append_axes("left", size=1.5, pad=0.5, sharey=ax1)
        ax2.fill_betweenx(self.x_d_flatten, 0, self.density, color='grey', alpha=0.5)
        ax2.yaxis.set_major_formatter(NullFormatter())  # Remove y-tick labels
        ax2.set_xlabel('Density')
        ax2.yaxis.tick_right()
        max_density = np.max(self.density)  # find the maximum density value
        ax2.set_xlim(max_density, 0)
        ax2.tick_params(axis='y', which='both', left=False, right=False)
        ax2.set_ylim(ax1.get_ylim())  # Ensure the density plot aligns correctly

        # Adding a legend to the plot that includes the quantile information
        ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Quantiles")
        plt.tight_layout()
        # save the plot to root_path/images
        plt.savefig(path + "kernel_sax_with_quantiles.png", dpi=300)
        print(f"Plot saved to {path}")
