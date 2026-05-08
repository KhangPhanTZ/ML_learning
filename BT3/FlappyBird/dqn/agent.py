"""DQN agent: replay memory, epsilon-greedy policy, target network, Huber loss."""

from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayMemory:
    """Standard FIFO replay buffer backed by collections.deque."""

    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Vanilla DQN with a target network and linear epsilon decay."""

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 2,
        hidden_dim: int = 24,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 30_000,
        memory_size: int = 50_000,
        batch_size: int = 64,
        target_update_steps: int = 1_000,
        device: Optional[torch.device] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.policy_net = DQN(state_dim, hidden_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, hidden_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # Huber loss: more robust to the +1 / -1 reward outliers than plain MSE.
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = ReplayMemory(memory_size)
        self.train_steps = 0

    # ---------------------------------------------------------------- policy
    @property
    def epsilon(self) -> float:
        """Linearly decayed exploration rate over `eps_decay_steps` gradient updates."""
        progress = min(self.train_steps / max(1, self.eps_decay_steps), 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * progress

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Return an action index. greedy=True forces pure exploitation (eps=0)."""
        if (not greedy) and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    def push(self, state, action, reward, next_state, done) -> None:
        self.memory.push(state, action, reward, next_state, done)

    # --------------------------------------------------------------- learning
    def train_step(self) -> Optional[float]:
        """Sample a batch and run one Bellman update; returns the loss (or None)."""
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states = torch.as_tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device)

        # Q(s, a) under current policy network.
        q_pred = self.policy_net(states).gather(1, actions).squeeze(1)

        # Bellman target r + gamma * max_a' Q_target(s', a')   (zeroed at terminal states).
        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1).values
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip to prevent rare large gradients from spiking the tiny network.
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------ I/O
    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = False) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
            self.policy_net.load_state_dict(ckpt["policy_state_dict"])
            self.target_net.load_state_dict(ckpt.get("target_state_dict", ckpt["policy_state_dict"]))
            if load_optimizer and "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.train_steps = ckpt.get("train_steps", 0)
        else:
            # Fallback for a raw state_dict checkpoint.
            self.policy_net.load_state_dict(ckpt)
            self.target_net.load_state_dict(ckpt)
