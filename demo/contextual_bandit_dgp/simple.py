"""Simple contextual bandit data generating process with linear reward structure."""

import numpy as np
from numpy.typing import NDArray

from .base import DGPBase


class Simple(DGPBase):
    """
    A simple contextual bandit DGP with linear reward functions.

    This class implements a contextual bandit data generating process where:
    - Context features are drawn from a standard normal distribution
    - Rewards are linear functions of the context features
    - Each arm has its own coefficient vector (beta)

    Attributes:
        n_features (int): Number of context features.
        n_arms (int): Number of available arms/actions.
        shared_weight (float): Weight for the shared component in [0, 1].
            0 = purely arm-specific, 1 = purely shared.
        beta (NDArray[np.float64]): Coefficient matrix of shape (n_arms, n_features + 1),
            where the first column represents intercepts.

    """

    def __init__(self, n_arms: int, n_features: int, shared_weight: float = 0.0) -> None:
        """
        Initialize the Simple contextual bandit DGP.

        Args:
            n_arms: Number of available arms/actions.
            n_features: Number of context features.
            shared_weight: Weight for the shared component in [0, 1].
                0.0 means purely arm-specific coefficients,
                1.0 means purely shared coefficients across all arms,
                values in between create a mixture. Defaults to 0.0.

        """
        self.n_features: int = n_features
        self.n_arms: int = n_arms
        self.shared_weight: float = shared_weight
        self.beta: NDArray[np.float64]
        self.setup()

    def setup(self) -> None:
        """
        Initialize the coefficient matrix for the reward functions.

        Generates a coefficient matrix as a weighted combination of:
        - A shared component (same across all arms)
        - Arm-specific components (unique to each arm)

        The final beta is computed as:
            beta = shared_weight * shared_beta + (1 - shared_weight) * arm_specific_beta

        The matrix has shape (n_arms, n_features + 1) to include intercepts.
        """
        # Shared component (broadcast to all arms)
        shared_beta = np.random.normal(size=(self.n_features + 1,))

        # Arm-specific components
        arm_specific_beta = np.random.normal(size=(self.n_arms, self.n_features + 1))

        # Weighted combination
        self.beta = self.shared_weight * shared_beta[np.newaxis, :] + (1 - self.shared_weight) * arm_specific_beta

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

        The reward is computed as a linear function: beta[a] @ [1, x],
        where the coefficient vector includes an intercept term.

        Args:
            a: Array of arm/action indices of shape (n_samples,).
            x: Context feature array of shape (n_samples, n_features).

        Returns:
            Array of rewards of shape (n_samples,).

        """
        # Augment x with intercept column
        x_augmented = np.hstack((np.ones((x.shape[0], 1)), x))

        # Get beta coefficients for each arm and compute dot product
        # beta[a] has shape (n_samples, n_features + 1)
        # x_augmented has shape (n_samples, n_features + 1)
        return np.sum(self.beta[a] * x_augmented, axis=1)

    def r_star(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the optimal (maximum) reward for given contexts.

        Evaluates the reward function for all arms and returns the maximum
        reward achievable for each context.

        Args:
            x: Context feature array of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,) representing the maximum reward
            across all arms for each context.

        """
        # Augment x with intercept column
        x_augmented = np.hstack((np.ones((x.shape[0], 1)), x))

        # Compute rewards for all arms and contexts
        # self.beta has shape (n_arms, n_features + 1)
        # x_augmented has shape (n_samples, n_features + 1)
        # all_rewards has shape (n_samples, n_arms)
        all_rewards = x_augmented @ self.beta.T

        # Return max across arms (axis=1) for each sample
        return np.max(all_rewards, axis=1)

    def a_star(self, x: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Compute optimal arm indices for given contexts.

        Evaluates the reward function for all arms and returns the arm index
        that maximizes reward for each context.

        Args:
            x: Context feature array of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,) containing the optimal arm index
            for each context.

        """
        # Augment x with intercept column
        x_augmented = np.hstack((np.ones((x.shape[0], 1)), x))

        # Compute rewards for all arms and contexts
        # self.beta has shape (n_arms, n_features + 1)
        # x_augmented has shape (n_samples, n_features + 1)
        # all_rewards has shape (n_samples, n_arms)
        all_rewards = x_augmented @ self.beta.T

        # Return argmax across arms (axis=1) for each sample
        return np.argmax(all_rewards, axis=1).astype(np.int_)
