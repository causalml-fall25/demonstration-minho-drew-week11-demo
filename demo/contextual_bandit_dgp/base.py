"""Base class for contextual bandit data generating processes."""

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.typing import NDArray


class DGPBase(metaclass=ABCMeta):
    """
    Abstract base class for contextual bandit data generating processes.

    Defines the interface for generating contexts, computing rewards,
    and finding optimal rewards in contextual bandit settings.
    """

    @abstractmethod
    def setup(self) -> None:
        """Initialize the data generating process parameters."""
        raise NotImplementedError

    def __init__(self, **kwargs) -> None:
        """
        Initialize the contextual bandit DGP.

        Args:
            **kwargs: Implementation-specific keyword arguments.

        """
        self.setup()

    @abstractmethod
    def x(self, num_units: int = 1) -> NDArray[np.float64]:
        """
        Generate context features.

        Args:
            num_units: Number of units/samples to generate.

        Returns:
            Array of context features.

        """
        raise NotImplementedError

    @abstractmethod
    def r(self, a: NDArray[np.int_], x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute rewards for given arms and contexts.

        Args:
            a: Array of arm/action indices.
            x: Array of context features.

        Returns:
            Array of rewards.

        """
        raise NotImplementedError

    @abstractmethod
    def r_star(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute optimal rewards for given contexts.

        Args:
            x: Array of context features.

        Returns:
            Array of optimal (maximum) rewards across all arms.

        """
        raise NotImplementedError

    @abstractmethod
    def a_star(self, x: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Compute optimal arm indices for given contexts.

        Args:
            x: Array of context features.

        Returns:
            Array of optimal arm indices that maximize rewards.

        """
        raise NotImplementedError
