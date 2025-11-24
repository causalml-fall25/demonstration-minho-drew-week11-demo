from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray


class DQN(nn.Module):
    """
    A Deep Q-Network (DQN) implementation for reinforcement learning.

    This class defines the architecture of a Deep Q-Network used for approximating Q-values in
    reinforcement learning tasks. It includes methods for forward passes, action selection using
    an epsilon-greedy policy, and updating Q-values using the Bellman equation through gradient descent.

    Attributes:
        n_features (int): Number of input features (state space size).
        n_actions (int): Number of possible actions in the environment.
        epsilon (float): The probability of selecting a random action (exploration vs. exploitation).
        optimizer (torch.optim.Adam): Optimizer used for updating the network parameters.
        affine1 (torch.nn.Linear): Fully connected layer to transform input features.
        q_value_head (torch.nn.Linear): Fully connected layer to output Q-values for each action.

    """

    def __init__(self, n_features: int, n_actions: int, lr: float = 3e-2, n_hidden_units: Union[int, Sequence[int]] = 128, epsilon: float = 0.1):
        """
        Initialize a DQN (Deep Q-Network) model.

        Args:
            n_features (int): Number of input features (size of the state space).
            n_actions (int): Number of possible actions in the environment.
            n_hidden_units (int): Number of hidden units in the network.
            lr (float, optional): Learning rate for the optimizer. Default is 3e-2.
            epsilon (float, optional): Probability of choosing a random action for epsilon-greedy policy. Default is 0.1.

        """
        super(DQN, self).__init__()

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

        self.q_value_head = nn.Linear(in_dim, n_actions)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Set epsilon
        self.epsilon = epsilon

        # Store number of actions
        self.n_actions = n_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_actions), representing Q-values for each action.

        """
        x = self.shared(x)
        q_values = self.q_value_head(x)
        return q_values

    def select_actions(self, state: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Select actions based on the current state using epsilon-greedy policy.

        Args:
            state (NDArray[np.float64]): Input state(s) of shape (batch_size, n_features).

        Returns:
            NDArray[np.int_]: Array of selected actions of shape (batch_size,).

        """
        # Convert state to tensor
        state_tensor = torch.from_numpy(state).float()

        # Get Q-values from forward pass
        q_values = self.forward(state_tensor)

        # Epsilon-greedy policy: Random action with probability epsilon
        if np.random.rand() < self.epsilon:
            actions = torch.randint(0, self.n_actions, (state_tensor.size(0),))
        else:
            actions = torch.argmax(q_values, dim=1)

        return actions.numpy().astype(np.int_)

    def finish_batch(
        self,
        rewards: NDArray[np.float64],
        actions: NDArray[np.int_],
        states: NDArray[np.float64],
    ) -> float:
        """
        Perform a single update to the Q-value estimates using the rewards and actions.

        Args:
            rewards (NDArray[np.float64]): Rewards for each state-action pair, shape (batch_size,).
            actions (NDArray[np.int_]): Selected actions for each state, shape (batch_size,).
            states (NDArray[np.float64]): States corresponding to each action, shape (batch_size, n_features).

        Returns:
            float: The computed loss value after the backward pass.

        """
        # Convert rewards, actions, and states to tensors
        rewards_tensor = torch.from_numpy(rewards).float()
        actions_tensor = torch.from_numpy(actions).int()
        states_tensor = torch.from_numpy(states).float()

        # Get Q-values for the selected actions
        current_q_values = self.forward(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze()

        # Compute the loss
        loss = nn.MSELoss()(current_q_values, rewards_tensor)

        # Zero the gradients, backpropagate, and update the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
