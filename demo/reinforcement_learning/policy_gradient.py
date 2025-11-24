from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from numpy.typing import NDArray


class PolicyGradient(nn.Module):
    """Policy gradient model for reinforcement learning."""

    def __init__(self, n_features: int, n_actions: int, n_hidden_units: Union[int, Sequence[int]] = 128, lr: float = 3e-2):
        """
        Initialize the policy network.

        Args:
            n_features: Number of input features.
            n_actions: Number of possible actions.
            n_hidden_units (int | Sequence[int], optional):
                Size(s) of hidden layers in the shared part of the network.
                If an int is given, a single hidden layer is used. If a sequence
                is given, one hidden layer is created per element. Defaults to 128.
            lr: Learning rate for the optimizer.

        """
        super(PolicyGradient, self).__init__()

        # Turn single int into a list, keep lists/tuples as-is
        if isinstance(n_hidden_units, int):
            hidden_sizes = [n_hidden_units]
        else:
            hidden_sizes = list(n_hidden_units)

        assert len(hidden_sizes) > 0, "n_hidden_units must contain at least one hidden layer size."

        # Build shared hidden layers: n_features -> h1 -> h2 -> ... -> hN
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        self.shared = nn.Sequential(*layers)

        # Define action head
        self.action_head = nn.Linear(in_dim, n_actions)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (n, n_features).

        Returns:
            Log probabilities of shape (n, n_actions).

        """
        x = self.shared(x)
        log_probs = f.log_softmax(self.action_head(x), dim=-1)
        return log_probs

    def select_actions(self, state: NDArray[np.float64]) -> tuple[NDArray[np.int_], torch.Tensor]:
        """
        Select actions based on the current states.

        Args:
            state: Input states of shape (n, n_features).

        Returns:
            Tuple of (actions, log_probs) where actions has shape (n,)
            and log_probs has shape (n, n_actions).

        """
        state_tensor = torch.from_numpy(state).float()
        log_probs = self.forward(state_tensor)

        # Convert to probabilities and sample
        probs = torch.exp(log_probs)
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return actions.detach().numpy().astype(np.int_), log_probs

    def finish_batch(
        self,
        rewards: NDArray[np.float64],
        actions: NDArray[np.int_],
        log_probs: torch.Tensor,
    ) -> float:
        """
        Calculate the policy loss and perform backpropagation.

        Args:
            rewards: Rewards of shape (n,).
            actions: Selected actions of shape (n,).
            log_probs: Log probabilities of shape (n, n_actions).

        Returns:
            The loss value as a float.

        """
        # Convert rewards to tensor
        rewards_tensor = torch.from_numpy(rewards).float()

        # Get log probabilities of the selected actions
        actions_tensor = torch.from_numpy(actions).long()
        selected_log_probs = log_probs[torch.arange(len(actions)), actions_tensor]

        # Calculate policy loss: -log_prob * reward
        policy_loss = -(selected_log_probs * rewards_tensor).mean()

        # Perform backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item()
