"""
IHDP-based contextual bandit DGP with semi-synthetic response surfaces.

Based on the Infant Health and Development Program (IHDP) dataset, adapted from:
https://github.com/ddimmery/softblock/blob/master/dgp/ihdp.py
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .base import DGPBase


class IHDP(DGPBase):
    """
    A contextual bandit DGP based on the IHDP dataset.

    This class implements a semi-synthetic data generating process using
    real covariates from the Infant Health and Development Program study.
    Response surfaces are generated synthetically based on these real covariates.

    The DGP supports multiple arms (treatments), extending the original binary
    treatment setting. Each arm has its own response surface.

    Attributes:
        n_features (int): Number of context features (25 IHDP covariates).
        n_arms (int): Number of available arms/actions.
        n_units (int): Number of units in the dataset.
        tau (float): Base treatment effect.
        sigma_y (float): Noise standard deviation for outcomes.
        contexts (NDArray[np.float64]): Preprocessed covariate matrix.
        mu (NDArray[np.float64]): Mean potential outcomes for each arm.

    """

    # IHDP covariate names
    COVARIATES = [
        "bw",
        "b.head",
        "preterm",
        "birth.o",
        "nnhealth",
        "momage",
        "sex",
        "twin",
        "b.marr",
        "mom.lths",
        "mom.hs",
        "mom.scoll",
        "cig",
        "first",
        "booze",
        "drugs",
        "work.dur",
        "prenatal",
        "ark",
        "ein",
        "har",
        "mia",
        "pen",
        "tex",
        "was",
    ]

    def __init__(
        self,
        n_arms: int = 2,
        tau: float = 4.0,
        sigma_y: float = 1.0,
        setting: str = "A",
        csv_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the IHDP contextual bandit DGP.

        Args:
            n_arms: Number of treatment arms. Defaults to 2.
            tau: Base treatment effect magnitude. Defaults to 4.0.
            sigma_y: Standard deviation of outcome noise. Defaults to 1.0.
            setting: Response surface setting, either 'A' (sparse linear) or
                'B' (exponential). Defaults to 'A'.
            csv_path: Path to the IHDP CSV file. If None, uses the bundled data.

        """
        self.n_arms: int = n_arms
        self.tau: float = tau
        self.sigma_y: float = sigma_y
        self.setting: str = setting

        # Load data
        if csv_path is None:
            csv_path = Path(__file__).parent / "ihdp.csv"
        self._load_data(csv_path)

        self.n_features: int = self.contexts.shape[1]
        self.n_units: int = self.contexts.shape[0]

        self.mu: NDArray[np.float64]
        self.setup()

    def _load_data(self, csv_path: str | Path) -> None:
        """
        Load and preprocess the IHDP dataset.

        Args:
            csv_path: Path to the IHDP CSV file.

        """
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Filter rows (following original: exclude non-white treated)
        ok_rows = np.logical_or(df["momwhite"] != 0, df["treat"] != 1)
        x = df.loc[ok_rows, self.COVARIATES].values.astype(np.float64)

        # Standardize continuous columns (those with more than 2 unique values)
        for col in range(x.shape[1]):
            if len(np.unique(x[:, col])) > 2:
                x[:, col] = (x[:, col] - x[:, col].mean()) / x[:, col].std()

        self.contexts = x

    def setup(self) -> None:
        """
        Initialize the response surfaces for each arm.

        Generates coefficient vectors for each arm and computes mean potential
        outcomes at each context point.
        """
        n_covs = self.contexts.shape[1]

        # Add intercept for coefficient generation
        x_with_intercept = np.hstack((np.ones((self.n_units, 1)), self.contexts))
        n_params = x_with_intercept.shape[1]

        # Generate coefficients for each arm
        self.mu = np.zeros((self.n_units, self.n_arms))

        for arm in range(self.n_arms):
            if self.setting == "A":
                # Sparse integer coefficients
                beta = np.random.choice([0, 1, 2, 3, 4], n_params, replace=True, p=[0.5, 0.2, 0.15, 0.1, 0.05])
                # Add arm-specific variation
                w = np.zeros_like(x_with_intercept)
                if arm > 0:
                    w = np.random.normal(0, 0.5, x_with_intercept.shape)
                self.mu[:, arm] = (x_with_intercept + w) @ beta

            elif self.setting == "B":
                # Continuous coefficients with exponential transformation
                beta = np.random.choice([0, 0.1, 0.2, 0.3, 0.4], n_params, replace=True, p=[0.6, 0.1, 0.1, 0.1, 0.1])
                w_val = 0.5 * arm  # Arm-specific shift
                w = np.ones_like(x_with_intercept) * w_val
                self.mu[:, arm] = np.exp((x_with_intercept + w) @ beta)

        # Adjust so that arm differences have mean tau (for arm 1 vs arm 0)
        if self.n_arms >= 2:
            adjustment = np.mean(self.mu[:, 1] - self.mu[:, 0]) - self.tau
            self.mu[:, 1:] -= adjustment

        # Center each arm's outcomes
        for arm in range(self.n_arms):
            self.mu[:, arm] = np.mean(self.mu[:, arm]) + (self.mu[:, arm] - np.mean(self.mu[:, arm]))

    def x(self, num_units: int = 1) -> NDArray[np.float64]:
        """
        Sample context features from the dataset.

        Args:
            num_units: Number of units/samples to generate.

        Returns:
            Array of shape (num_units, n_features) containing sampled contexts.

        """
        indices = np.random.choice(self.n_units, size=num_units, replace=True)
        return self.contexts[indices]

    def r(self, a: NDArray[np.int_], x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute rewards for given arms and contexts.

        Finds the nearest context in the dataset and returns the corresponding
        potential outcome plus noise.

        Args:
            a: Array of arm/action indices of shape (n_samples,).
            x: Context feature array of shape (n_samples, n_features).

        Returns:
            Array of rewards of shape (n_samples,).

        """
        # Find nearest neighbors in the dataset for each query context
        indices = self._find_nearest(x)

        # Get mean outcomes for selected arms
        mean_rewards = self.mu[indices, a]

        # Add noise
        return mean_rewards + np.random.normal(0, self.sigma_y, size=len(a))

    def r_star(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute optimal rewards for given contexts.

        Args:
            x: Context feature array of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,) representing the maximum expected
            reward across all arms.

        """
        indices = self._find_nearest(x)
        return np.max(self.mu[indices], axis=1)

    def a_star(self, x: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Compute optimal arm indices for given contexts.

        Args:
            x: Context feature array of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,) containing the optimal arm index.

        """
        indices = self._find_nearest(x)
        return np.argmax(self.mu[indices], axis=1).astype(np.int_)

    def _find_nearest(self, x: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Find indices of nearest contexts in the dataset.

        Args:
            x: Query contexts of shape (n_samples, n_features).

        Returns:
            Array of indices into self.contexts.

        """
        # Compute squared distances to all dataset points
        # x: (n_query, n_features), self.contexts: (n_units, n_features)
        # Using broadcasting: (n_query, 1, n_features) - (1, n_units, n_features)
        diff = x[:, np.newaxis, :] - self.contexts[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2)
        return np.argmin(sq_dist, axis=1)

    def sample_batch(self, batch_size: int) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        """
        Sample a batch of contexts with their indices.

        Useful for getting contexts along with their dataset indices for
        direct potential outcome lookup.

        Args:
            batch_size: Number of samples to draw.

        Returns:
            Tuple of (contexts, indices) where contexts has shape
            (batch_size, n_features) and indices has shape (batch_size,).

        """
        indices = np.random.choice(self.n_units, size=batch_size, replace=True)
        return self.contexts[indices], indices

    def r_direct(self, a: NDArray[np.int_], indices: NDArray[np.int_]) -> NDArray[np.float64]:
        """
        Compute rewards using dataset indices directly (faster than r()).

        Args:
            a: Array of arm/action indices of shape (n_samples,).
            indices: Dataset indices from sample_batch().

        Returns:
            Array of rewards of shape (n_samples,).

        """
        mean_rewards = self.mu[indices, a]
        return mean_rewards + np.random.normal(0, self.sigma_y, size=len(a))
