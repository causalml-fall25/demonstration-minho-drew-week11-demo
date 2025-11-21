from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from numpy.typing import NDArray
from torch.distributions import Categorical

# Named tuple to store actions and their log probabilities, and state values
SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


class ActorCritic(nn.Module):
    """Implements both actor and critic in one model."""

    def __init__(self, n_features: int, gamma: float = 0.99, lr: float = 3e-2):
        """Initialize the actor and critic networks."""
        super(ActorCritic, self).__init__()

        # Define the layers of the network
        self.affine1 = nn.Linear(n_features, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        # Buffers to store actions and rewards during the episode
        self.saved_actions = []
        self.rewards = []

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Small epsilon to avoid numerical instability
        self.eps = np.finfo(np.float32).eps.item()

        # Discount factor for reward calculation
        self.gamma = gamma

    def forward(self, x: torch.Tensor):
        """Forward pass through the network for both actor and critic."""
        x = f.relu(self.affine1(x))
        action_prob = f.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_prob, state_value

    def select_action(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Select an action based on the current state.

        Args:
            state (NDArray[np.float64]): The input state.

        Returns:
            action (NDArray[np.float64]): The selected action.

        """
        state_tensor = torch.from_numpy(state).float()
        action_probs, state_value = self.forward(state_tensor)

        # Create a categorical distribution over the action probabilities
        m = Categorical(action_probs)

        # Sample an action from the distribution
        action = m.sample()

        # Save the action's log probability and the state value for later use
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return np.array([action.item()])

    def finish_episode(self):
        """Calculate the policy and value loss, then perform backpropagation."""
        r = 0
        policy_losses, value_losses = [], []
        returns = []

        # Calculate the discounted returns
        for reward in reversed(self.rewards):
            r = reward + self.gamma * r
            returns.insert(0, r)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        # Calculate the loss for each saved action
        for (log_prob, value), r in zip(self.saved_actions, returns):
            advantage = r - value.item()
            policy_losses.append(-log_prob * advantage)  # Actor loss
            value_losses.append(f.smooth_l1_loss(value, torch.tensor([r])))  # Critic loss

        # Perform the backpropagation
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()

        # Clear the rewards and actions after the episode
        self.rewards.clear()
        self.saved_actions.clear()
