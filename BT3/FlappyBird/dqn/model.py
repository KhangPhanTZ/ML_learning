"""DQN architecture: 3 -> 24 -> 24 -> 2 with ReLU on hidden layers."""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """Feed-forward Q-network.

    Layout (matches the BT3 spec):
        Input    : 3 features  -> [bird_y, top_pipe_y, bottom_pipe_y]   (normalised)
        Hidden 1 : Linear(3, 24)  + ReLU
        Hidden 2 : Linear(24, 24) + ReLU
        Output   : Linear(24, 2)  -- raw Q-values for [NOOP, FLAP]
    """

    def __init__(self, state_dim: int = 3, hidden_dim: int = 24, action_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)
