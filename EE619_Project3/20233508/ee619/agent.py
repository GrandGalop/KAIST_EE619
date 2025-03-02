"""Agent for DMControl Walker-Run task."""
from os.path import abspath, dirname, join, realpath
from typing import Dict, Tuple

from dm_env import TimeStep
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory


def flatten_and_concat(dmc_observation: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert a DMControl observation (OrderedDict of NumPy arrays)
    into a single NumPy array.

    """
    return np.concatenate([[obs] if np.isscalar(obs) else obs.ravel()
                           for obs in dmc_observation.values()])

def to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert NumPy array to a PyTorch Tensor of data type torch.float32"""
    return torch.as_tensor(array, dtype=torch.float32)

class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self) -> None:
        # Create class variables here if you need to.
        # Example:
        #     self.policy = torch.nn.Sequential(...)

        self.policy = Policy(24, 6)
        self.path = join(ROOT, 'trained_model.pt')

    def act(self, time_step: TimeStep) -> np.ndarray:
        
        """Returns the action to take for the current time-step.

        Args:
            time_step: a namedtuple with four fields step_type, reward,
                discount, and observation.
        """
        # You can access each member of time_step by time_step.[name], a
        # for example, time_step.reward or time_step.observation.
        # You may also check if the current time-step is the last one
        # of the episode, by calling the method time_step.last().
        # The return value will be True if it is the last time-step,
        # and False otherwise.
        # Note that the observation is given as an OrderedDict of
        # NumPy arrays, so you would need to convert it into a
        # single NumPy array before you feed it into your network.
        # It can be done by using the `flatten_and_concat` function, e.g.,
        #   observation = flatten_and_concat(time_step.observation)
        #   logits = self.policy(torch.as_tensor(observation))
        #return np.ones(6)

        observation = flatten_and_concat(time_step.observation)
        action = self.policy.act(observation)
        return np.tanh(action)
    
    def load(self):
        """Loads network parameters if there are any."""

        self.policy.load_state_dict(torch.load(self.path))

class Policy(nn.Module):
    """3-Layer MLP to use as a policy for DMControl environments."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.scale = nn.Parameter(torch.zeros(out_features)) # tensor.size([6])
        torch.nn.init.constant_(self.scale, -0.5) # tensor([-0.5, -0.5,-0.5,-0.5,-0.5,-0.5])

        self.critic = nn.Sequential(
                                    nn.Linear(in_features, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, out_features)
                                    )

    def forward(self, input: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the location and scale for the Gaussian distribution
        to sample the action from.

        """
        loc = torch.relu(self.fc1(input))
        loc = torch.relu(self.fc2(loc))
        loc = self.fc3(loc)
        scale = self.scale.exp().expand_as(loc)
        return loc, scale

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Sample an action for the given observation."""
        loc, scale = self.forward(to_tensor(observation).unsqueeze(0))

        # Normal (loc, scale): makes normal distribution / loc: means / scale : stdevs
        # Independent(base_distribution, reinterpreted_batch_ndims): 정규 분포로부터 독립적인(action-wise independent) 행동을 샘플링한다는 것을 나타냅니다. 이것은 각 행동 차원의 확률 분포가 서로 독립적으로 결정되는 것을 의미합니다.
        action = Independent(Normal(loc, scale), 1).sample().squeeze(0).numpy()

        return action