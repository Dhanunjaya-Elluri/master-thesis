#!/usr/bin/env python
# coding: utf-8

"""Lloyd-Max quantizer implementation."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import logging
import numpy as np
from scipy import integrate
from typing import Optional, Tuple, Callable

logger = logging.getLogger(__name__)


class LloydMaxQuantizer:
    def __init__(
        self,
        x: np.ndarray,
        density_func: Callable[[float], float],
        n_codewords: int,
        epochs: int = 100,
        verbose: bool = False,
        random_state: Optional[int] = None,
        init_codewords: str = "random",
    ):
        """
        Lloyd-Max quantizer implementation to find the optimal way to quantize this continuous data
        into discrete symbols, thereby improving generalization.

        Inspired from:
        https://github.com/MatthiasJakobs/tsx/blob/master/tsx/quantizers/prob_sax.py

        Args:
            x (np.ndarray): Input data.
            density_func (Callable[[float], float]): Density function.
            n_codewords (int): Number of codewords (quantization levels).
            epochs (int, optional): Number of training epochs. Defaults to 100.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.
            init_codewords (str, optional): Initialization method for codewords. Defaults to 'random'.
        """
        assert init_codewords in [
            "random",
        ], "Invalid Codewords initialization. Supported initializations: 'random'."

        self.x = x
        self.density_func = density_func
        self.n_codewords = n_codewords
        self.epochs = epochs
        self.verbose = verbose
        self.random_state = random_state
        self.init_codewords = init_codewords

    def fit(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the Lloyd-Max quantizer to the input data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the boundaries and codewords.
        """
        rng = np.random.RandomState(self.random_state)

        # Initialize codewords
        c_min, c_max = self.x.min(), self.x.max()
        codewords = self._initialize_codewords_random(c_min, c_max, rng)

        # Initialize boundaries
        boundaries = np.zeros((self.n_codewords + 1))
        boundaries[0], boundaries[-1] = self.x.min(), self.x.max()

        for epoch in range(self.epochs):
            boundaries_before = boundaries[1:-1].copy()
            codewords_before = codewords.copy()

            # Update boundaries based on codewords
            for j in range(1, self.n_codewords):
                boundaries[j] = 0.5 * (codewords[j - 1] + codewords[j])

            # Update codewords based on new boundaries
            for i in range(len(codewords)):
                bi = boundaries[i]
                bi_plus_1 = boundaries[i + 1]
                if bi == bi_plus_1:
                    codewords[i] = 0
                else:
                    numerator = integrate.quad(
                        lambda t: t * self.density_func(t), bi, bi_plus_1
                    )[0]
                    denominator = integrate.quad(self.density_func, bi, bi_plus_1)[0]
                    codewords[i] = (
                        numerator / denominator if denominator != 0 else 0
                    )  # Avoid division by zero

            # Compute delta and see if it decreases
            boundaries_delta = np.abs(boundaries[1:-1] - boundaries_before).mean()
            codewords_delta = np.abs(codewords - codewords_before).mean()

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}: Boundaries delta: {boundaries_delta:.10f}, "
                    f"Codewords delta: {codewords_delta:.10f}"
                )

            # Check stopping criteria
            if boundaries_delta == 0 and codewords_delta == 0:
                logger.info("Stopping criteria reached. Terminating training.")
                break

        return boundaries, codewords

    def _initialize_codewords_random(
        self, c_min: float, c_max: float, rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Initialize codewords randomly.

        Args:
            c_min (float): Minimum value for codewords initialization.
            c_max (float): Maximum value for codewords initialization.
            rng (np.random.RandomState): Random number generator.

        Returns:
            np.ndarray: Randomly initialized codewords.
        """
        return rng.uniform(c_min, c_max, size=self.n_codewords)
