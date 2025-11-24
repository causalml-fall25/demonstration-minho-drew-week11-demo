from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from numpy.typing import NDArray


class ActorCritic(nn.Module):
    """
    Actor-Critic network for reinforcement learning.

    This model combines both the actor (policy network) and the critic (value network) into a single model.
    The actor selects actions based on the policy, while the critic estimates the value of the current state.
    The model is trained using the policy gradient method where the actor updates its policy based on the
    advantage calculated from the critic's value estimation.

    Args:
        n_features (int): The number of features in the input state.
        n_actions (int): The number of possible actions the agent can take.
        n_hidden_units (int | Sequence[int], optional):
            Size(s) of hidden layers in the shared part of the network.
            If an int is given, a single hidden layer is used. If a sequence
            is given, one hidden layer is created per element. Defaults to 128.
        lr (float, optional): Learning rate for the optimizer. Defaults to 3e-2.

    """

    def __init__(self, n_features: int, n_actions: int, n_hidden_units: Union[int, Sequence[int]] = 128, lr: float = 3e-2):
        """Initialize the actor and critic networks."""
        super(ActorCritic, self).__init__()

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

        # Heads take input from last hidden layer (size = hidden_sizes[-1])
        self.action_head = nn.Linear(in_dim, n_actions)
        self.value_head = nn.Linear(in_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def actor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the actor network to compute the action probabilities.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, n_features), representing the state.

        Returns:
            torch.Tensor: The log probabilities of each action for the given state.

        """
        x = self.shared(x)
        log_probs = f.log_softmax(self.action_head(x), dim=-1)  # Compute the log-probabilities of each action
        return log_probs

    def critic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network to estimate the value of the current state.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, n_features), representing the state.

        Returns:
            torch.Tensor: The estimated value of the state.

        """
        x = self.shared(x)
        state_value = self.value_head(x)  # Compute the state value using the value head
        return state_value

    def select_actions(self, state: NDArray[np.float64]) -> tuple[NDArray[np.int_], torch.Tensor]:
        """
        Select an action based on the current state using the actor network.

        The action is selected by sampling from the policy distribution output by the actor network.

        Args:
            state (NDArray[np.float64]): The input state, an array of shape (n_features).

        Returns:
            tuple: A tuple containing:
                - action (NDArray[np.int_]): The selected action (a scalar integer).
                - log_probs (torch.Tensor): The log probabilities of each action.

        """
        state_tensor = torch.from_numpy(state).float()  # Convert state to tensor
        log_probs = self.actor(state_tensor)  # Get log-probabilities from the actor

        # Convert log-probs to probabilities and sample an action based on the distribution
        probs = torch.exp(log_probs)  # Convert log-probs to actual probabilities
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)  # Sample action

        return actions.detach().numpy().astype(np.int_), log_probs

    def finish_batch(
        self,
        rewards: NDArray[np.float64],
        states: NDArray[np.float64],
        actions: NDArray[np.int_],
        log_probs: torch.Tensor,
    ) -> float:
        """
        Calculate the policy loss (actor loss) and perform backpropagation.

        This function computes the advantage of each state-action pair, the policy loss, and updates the actor
        network's parameters using backpropagation. The critic's value estimate is used to compute the advantage.

        Args:
            rewards (NDArray[np.float64]): The rewards received from the environment for each action taken.
            states (NDArray[np.float64]): The states observed from the environment.
            actions (NDArray[np.int_]): The actions taken by the agent.
            log_probs (torch.Tensor): The log-probabilities of the selected actions.

        Returns:
            float: The computed actor loss.

        """
        # Convert rewards and states to tensors
        rewards_tensor = torch.tensor(rewards).float()  # Rewards as a tensor
        state_tensor = torch.from_numpy(states).float()  # States as a tensor
        actions_tensor = torch.from_numpy(actions).long()  # Actions as a tensor (long for indexing)

        # Get the log probabilities of the selected actions (indexed by actions_tensor)
        selected_log_probs = log_probs[torch.arange(len(actions)), actions_tensor]

        # Estimate state values from the critic
        state_value = self.critic(state_tensor)

        # Compute the advantage (difference between reward and state value)
        advantages = rewards_tensor - state_value.squeeze()  # Squeeze to remove extra dimension

        # Compute the actor loss (negative log-probability * advantage)
        actor_loss = -(selected_log_probs * advantages).mean()  # Mean over the batch
        critic_loss = nn.MSELoss()(state_value.squeeze(), rewards_tensor)  # Mean squared error loss

        loss = actor_loss + critic_loss
        # Backpropagate and update the network's parameters
        self.optimizer.zero_grad()  # Zero out any previous gradients
        loss.backward()  # Perform backpropagation
        self.optimizer.step()  # Update the parameters using the gradients

        return actor_loss.item()  # Return the actor loss as a float
