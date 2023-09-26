"""Lloyd-Max quantizer implementation."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"


import numpy as np
from scipy import integrate
from typing import Optional, Tuple, List, Callable


class LloydMaxQuantizer:
    def __init__(
        self,
        x: np.ndarray,
        density: Callable[[float], float],
        n_codewords: int,
        epochs: int = 100,
        verbose: bool = False,
        random_state: Optional[int] = None,
        init_codewords: str = "random",
    ):
        """
        Lloyd-Max quantizer implementation.
        Inspired from:
        https://github.com/MatthiasJakobs/tsx/blob/master/tsx/quantizers/prob_sax.py
        https://github.com/JosephChataignon/Max-Lloyd-algorithm/blob/master/1%20dimension/max_lloyd_1D.py

        Args:
            x (np.ndarray): Input data.
            density (Callable[[float], float]): Density function.
            n_codewords (int): Number of codewords (quantization levels).
            epochs (int, optional): Number of training epochs. Defaults to 100.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.
            init_codewords (str, optional): Initialization method for codewords. Either 'random' or 'kmeans++'.
                                            Defaults to 'random'.
        """
        assert init_codewords in [
            "random",
            "kmeans++",
        ], "Invalid Codewords initialization. Supported initializations: 'random', 'kmeans++'."

        self.x = x
        self.density = density
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
        c_min, c_max = self.x.min() - 1, self.x.max() + 1
        codewords = (
            self._initialize_codewords_kmeanspp(c_min, c_max, rng)
            if self.init_codewords == "kmeans++"
            else self._initialize_codewords_random(c_min, c_max, rng)
        )

        # Initialize boundaries
        boundaries = np.zeros((self.n_codewords + 1))
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf

        # Run for specified epochs
        for epoch in range(self.epochs):
            boundaries_before = boundaries[1:-1].copy()
            codewords_before = codewords.copy()

            # Calculate new boundaries (cutlines)
            for j in range(1, self.n_codewords):
                boundaries[j] = 0.5 * (codewords[j - 1] + codewords[j])

            # Calculate new codewords (centroids)
            for i in range(len(codewords)):
                bi = boundaries[i]
                biplus1 = boundaries[i + 1]
                if bi == biplus1:
                    codewords[i] = 0
                else:
                    numerator, _ = integrate.quad(
                        lambda t: t * self.density(t), boundaries[i], boundaries[i + 1]
                    )
                    denominator, _ = integrate.quad(
                        self.density, boundaries[i], boundaries[i + 1]
                    )
                    codewords[i] = numerator / denominator

            # Compute delta and see if it decreases
            boundaries_delta = np.abs(boundaries[1:-1] - boundaries_before).mean()
            codewords_delta = np.abs(codewords - codewords_before).mean()

            if self.verbose:
                print(
                    "Epoch:",
                    epoch,
                    "Boundaries Delta:",
                    boundaries_delta,
                    "Codewords Delta:",
                    codewords_delta,
                )

            # Check stopping criteria
            if boundaries_delta == 0 and codewords_delta == 0:
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

    def _initialize_codewords_kmeanspp(
        self, c_min: float, c_max: float, rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Initialize codewords using k-means++ algorithm.

        Args:
            c_min (float): Minimum value for codewords initialization.
            c_max (float): Maximum value for codewords initialization.
            rng (np.random.RandomState): Random number generator.

        Returns:
            np.ndarray: Codewords initialized using k-means++.
        """
        codewords = np.zeros(self.n_codewords)
        codewords[0] = rng.uniform(c_min, c_max)

        for j in range(1, self.n_codewords):
            distances = self._calculate_distances(self.x, codewords[:j])
            probabilities = self._calculate_probabilities(distances)
            codewords[j] = self._sample_new_codeword(self.x, probabilities, rng)

        return codewords

    @staticmethod
    def _calculate_distances(x: np.ndarray, codewords: np.ndarray) -> np.ndarray:
        """
        Calculate distances between input data and codewords.

        Args:
            x (np.ndarray): Input data.
            codewords (np.ndarray): Codewords.

        Returns:
            np.ndarray: Distances between input data and codewords.
        """
        # Specify the data type explicitly
        distances = np.zeros_like(x, dtype=np.float64)
        for i in range(len(codewords)):
            distances += np.abs(x - codewords[i])
        return distances
        # distances = np.abs(x[:, np.newaxis] - codewords)  # Calculate absolute differences
        # return distances

    @staticmethod
    def _calculate_probabilities(distances: np.ndarray) -> np.ndarray:
        """
        Calculate probabilities based on distances.

        Args:
            distances (np.ndarray): Distances between input data and codewords.

        Returns:
            np.ndarray: Probabilities based on distances.
        """
        return 1.0 / distances

    @staticmethod
    def _sample_new_codeword(
        x: np.ndarray, probabilities: np.ndarray, rng: np.random.RandomState
    ) -> float:
        """
        Sample a new codeword based on probabilities.

        Args:
            x (np.ndarray): Input data.
            probabilities (np.ndarray): Probabilities.
            rng (np.random.RandomState): Random number generator.

        Returns:
            float: Sampled new codeword.
        """
        normalized_probabilities = probabilities / np.sum(probabilities)
        return rng.choice(x, p=normalized_probabilities)
