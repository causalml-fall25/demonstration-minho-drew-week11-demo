"""Complex contextual bandit DGP with Gaussian Process response surfaces."""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from .base import DGPBase


class Complex(DGPBase):
    """
    A contextual bandit DGP with Gaussian Process response surfaces.

    This class implements a contextual bandit data generating process where:
    - Context features are drawn from a standard normal distribution
    - Rewards are sampled from a GP with an RBF kernel, providing nonlinear surfaces
    - Each arm has its own GP-sampled reward function

    The GP is approximated by sampling function values at inducing points and
    interpolating using the RBF kernel.

    Attributes:
        n_features (int): Number of context features.
        n_arms (int): Number of available arms/actions.
        length_scale (float): Length scale parameter for the RBF kernel.
        amplitude (float): Amplitude (output scale) for the GP.
        n_inducing (int): Number of inducing points for GP approximation.
        inducing_points (NDArray[np.float64]): Inducing point locations.
        inducing_values (NDArray[np.float64]): GP function values at inducing points.

    """

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        length_scale: float = 1.0,
        amplitude: float = 1.0,
        n_inducing: int = 100,
    ) -> None:
        """
        Initialize the Complex contextual bandit DGP.

        Args:
            n_arms: Number of available arms/actions.
            n_features: Number of context features.
            length_scale: Length scale for the RBF kernel. Larger values create
                smoother reward surfaces. Defaults to 1.0.
            amplitude: Output scale for the GP. Controls the magnitude of
                reward variations. Defaults to 1.0.
            n_inducing: Number of inducing points for GP approximation.
                More points give better approximation but slower computation.
                Defaults to 100.

        """
        self.n_features: int = n_features
        self.n_arms: int = n_arms
        self.length_scale: float = length_scale
        self.amplitude: float = amplitude
        self.n_inducing: int = n_inducing
        self.inducing_points: NDArray[np.float64]
        self.inducing_values: NDArray[np.float64]
        self.setup()

    def _rbf_kernel(self, x1: NDArray[np.float64], x2: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the RBF (squared exponential) kernel between two sets of points.

        Args:
            x1: First set of points, shape (n1, n_features).
            x2: Second set of points, shape (n2, n_features).

        Returns:
            Kernel matrix of shape (n1, n2).

        """
        sq_dist = cdist(x1, x2, metric="sqeuclidean")
        return self.amplitude**2 * np.exp(-0.5 * sq_dist / self.length_scale**2)

    def setup(self) -> None:
        """
        Initialize the GP by sampling function values at inducing points.

        Generates inducing points from a standard normal distribution and
        samples GP function values at these points for each arm.
        """
        # Sample inducing points from the context distribution
        self.inducing_points = np.random.normal(size=(self.n_inducing, self.n_features))

        # Compute kernel matrix at inducing points
        kernel = self._rbf_kernel(self.inducing_points, self.inducing_points)

        # Add jitter for numerical stability
        kernel += 1e-6 * np.eye(self.n_inducing)

        # Cholesky decomposition for sampling
        chol = np.linalg.cholesky(kernel)

        # Sample GP function values for each arm
        # Shape: (n_arms, n_inducing)
        standard_normal = np.random.normal(size=(self.n_arms, self.n_inducing))
        self.inducing_values = standard_normal @ chol.T

    def _predict_gp(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict GP mean at new points using inducing point approximation.

        Uses kernel interpolation to predict function values at new points
        based on the sampled values at inducing points.

        Args:
            x: Query points, shape (n_samples, n_features).

        Returns:
            Predicted values for each arm, shape (n_samples, n_arms).

        """
        # Kernel between query points and inducing points
        kernel_query = self._rbf_kernel(x, self.inducing_points)

        # Kernel at inducing points (with jitter)
        kernel_inducing = self._rbf_kernel(self.inducing_points, self.inducing_points)
        kernel_inducing += 1e-6 * np.eye(self.n_inducing)

        # Solve for weights: kernel_inducing @ weights = inducing_values.T
        weights = np.linalg.solve(kernel_inducing, self.inducing_values.T)

        # Predict: (n_samples, n_inducing) @ (n_inducing, n_arms) -> (n_samples, n_arms)
        return kernel_query @ weights

    def x(self, num_units: int = 1) -> NDArray[np.float64]:
        """
        Generate context features.

        Args:
            num_units: Number of units/samples to generate.

        Returns:
            Array of shape (num_units, n_features) containing context features
            drawn from a standard normal distribution.

        """
        return np.random.normal(size=(num_units, self.n_features))

    def r(self, a: NDArray[np.int_], x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the reward for given arms and contexts.

        The reward is computed by evaluating the GP-sampled reward surface
        for the selected arm at each context.

        Args:
            a: Array of arm/action indices of shape (n_samples,).
            x: Context feature array of shape (n_samples, n_features).

        Returns:
            Array of rewards of shape (n_samples,).

        """
        # Get GP predictions for all arms: (n_samples, n_arms)
        all_rewards = self._predict_gp(x)

        # Select rewards for the chosen arms
        return all_rewards[np.arange(len(a)), a]

    def r_star(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the optimal (maximum) reward for given contexts.

        Evaluates the GP reward function for all arms and returns the maximum
        reward achievable for each context.

        Args:
            x: Context feature array of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,) representing the maximum reward
            across all arms for each context.

        """
        # Get GP predictions for all arms: (n_samples, n_arms)
        all_rewards = self._predict_gp(x)

        # Return max across arms
        return np.max(all_rewards, axis=1)

    def a_star(self, x: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Compute optimal arm indices for given contexts.

        Evaluates the GP reward function for all arms and returns the arm index
        that maximizes reward for each context.

        Args:
            x: Context feature array of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,) containing the optimal arm index
            for each context.

        """
        # Get GP predictions for all arms: (n_samples, n_arms)
        all_rewards = self._predict_gp(x)

        # Return argmax across arms
        return np.argmax(all_rewards, axis=1).astype(np.int_)
