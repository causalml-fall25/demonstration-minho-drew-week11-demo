import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from numpy.typing import NDArray


class PolicyGradient(nn.Module):
    """Policy gradient model for reinforcement learning."""

    def __init__(self, n_features: int, n_actions: int, n_hidden_units: int = 128, lr: float = 3e-2):
        """
        Initialize the policy network.

        Args:
            n_features: Number of input features.
            n_actions: Number of possible actions.
            n_hidden_units: Number of hidden units.
            lr: Learning rate for the optimizer.

        """
        super(PolicyGradient, self).__init__()

        # Define the layers of the network
        self.affine1 = nn.Linear(n_features, n_hidden_units)
        self.action_head = nn.Linear(n_hidden_units, n_actions)

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
        x = f.relu(self.affine1(x))
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
