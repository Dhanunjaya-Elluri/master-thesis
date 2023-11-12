#!/usr/bin/env python
# coding: utf-8

"""Lloyd-Max quantizer implementation."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import numpy as np
from scipy import integrate
from typing import Optional, Callable, Tuple

# constant for initialization method
INIT_RANDOM = "random"


class LloydMaxQuantizer:
    def __init__(
        self,
        density_func: Callable[[float], float],
        n_codewords: int,
        random_state: Optional[int] = None,
        init_codewords: str = "random",
        verbose: bool = True,
    ):
        self.x = None
        self.density_func = density_func
        self.n_codewords = n_codewords
        self.random_state = random_state
        self.init_codewords = init_codewords
        self.verbose = verbose
        self.boundaries = None
        self.codewords = None
        self._validate_parameters()

    def _validate_parameters(self):
        supported_initializations = [INIT_RANDOM]
        assert (
            self.init_codewords in supported_initializations
        ), f"Invalid initialization method. Supported initializations: {supported_initializations}."

    def fit(self, x: np.ndarray, epochs: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the quantizer to the input data.

        Args:
            x (np.ndarray): Input data.
            epochs (int): Number of epochs to train the quantizer.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Boundaries and codewords.
        """
        self.x = x
        rng = np.random.RandomState(self.random_state)

        # Initialize codewords and boundaries
        self.codewords = self._initialize_codewords(rng)
        self.boundaries = self._initialize_boundaries()

        for epoch in range(epochs):
            boundaries_before = self.boundaries[1:-1].copy()
            codewords_before = self.codewords.copy()

            # Update boundaries based on codewords
            self._update_boundaries()

            # Update codewords based on new boundaries
            self._update_codewords()

            # Check stopping criteria
            if self._has_converged(boundaries_before, codewords_before, epoch):
                print("Stopping criteria reached. Terminating training.")
                break

        return self.boundaries, self.codewords

    def _initialize_codewords(self, rng: np.random.RandomState) -> np.ndarray:
        """
        Initialize codewords randomly.
        Args:
            rng: Random number generator.

        Returns:
            np.ndarray: Randomly initialized codewords.
        """
        if self.init_codewords == INIT_RANDOM:
            c_min, c_max = self.x.min(), self.x.max()
            return rng.uniform(c_min, c_max, size=self.n_codewords)

        raise NotImplementedError(
            f"Initialization method {self.init_codewords} not implemented."
        )

    def _initialize_boundaries(self) -> np.ndarray:
        """
        Initialize boundaries.

        Returns:
            np.ndarray: Initialized boundaries.
        """
        boundaries = np.zeros((self.n_codewords + 1))
        boundaries[0], boundaries[-1] = self.x.min(), self.x.max()
        return boundaries

    def _update_boundaries(self) -> None:
        """
        Update boundaries based on current codewords.
        """

        for j in range(1, self.n_codewords):
            self.boundaries[j] = 0.5 * (self.codewords[j - 1] + self.codewords[j])

    def _update_codewords(self) -> None:
        """
        Update codewords based on current boundaries.
        """

        for i in range(len(self.codewords)):
            bi = self.boundaries[i]
            bi_plus_1 = self.boundaries[i + 1]
            if bi == bi_plus_1:
                self.codewords[i] = 0
            else:
                try:
                    numerator = integrate.quad(
                        lambda t: t * self.density_func(t), bi, bi_plus_1
                    )[0]
                    denominator = integrate.quad(self.density_func, bi, bi_plus_1)[0]
                except Exception as e:
                    print(f"Integration Failed: {e}")
                    raise e

            self.codewords[i] = numerator / denominator if denominator != 0 else 0

    def _has_converged(
        self, boundaries_before: np.ndarray, codewords_before: np.ndarray, epoch: int
    ) -> bool:
        """
        Check if the algorithm has converged.

        Args:
            boundaries_before (np.ndarray): Previous boundaries.
            codewords_before (np.ndarray): Previous codewords.

        Returns:
            bool: True if the algorithm has converged, False otherwise.
        """
        boundaries_delta = np.abs(self.boundaries[1:-1] - boundaries_before).mean()
        codewords_delta = np.abs(self.codewords - codewords_before).mean()

        if self.verbose:
            print(
                f"Epoch {epoch + 1}, Boundaries delta: {boundaries_delta:.10f}, Codewords delta: {codewords_delta:.10f}"
            )

        return boundaries_delta == 0 and codewords_delta == 0
