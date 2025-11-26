"""
Simple REINFORCE implementation for the Food Redistribution environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class ReinforceConfig:
    episodes: int = 500
    gamma: float = 0.99
    lr: float = 1e-3
    hidden_dim: int = 128
    batch_size: int = 5


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReinforceAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: ReinforceConfig) -> None:
        self.config = config
        self.policy = PolicyNet(obs_dim, action_dim, config.hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def train(self, env) -> Tuple[List[float], List[float]]:
        returns: List[float] = []
        batch_log_probs: List[torch.Tensor] = []
        batch_rewards: List[torch.Tensor] = []
        entropy_per_episode: List[float] = []

        for episode in range(1, self.config.episodes + 1):
            obs, _ = env.reset()
            done = False
            episode_rewards: List[float] = []
            episode_log_probs: List[torch.Tensor] = []
            episode_entropies: List[float] = []

            while not done:
                obs_tensor = torch.from_numpy(obs).float().to(self.device)
                action_probs = self.policy(obs_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                entropy = dist.entropy().item()
                episode_entropies.append(entropy)
                next_obs, reward, terminated, truncated, _ = env.step(action.item())

                episode_rewards.append(reward)
                episode_log_probs.append(dist.log_prob(action))
                obs = next_obs
                done = terminated or truncated

            discounted_returns = self._discount_rewards(episode_rewards)
            returns.append(sum(episode_rewards))

            if episode_entropies:
                mean_entropy = float(np.mean(episode_entropies))
                entropy_per_episode.append(mean_entropy)

            batch_log_probs.append(torch.stack(episode_log_probs))
            batch_rewards.append(torch.tensor(discounted_returns, dtype=torch.float32, device=self.device))

            if episode % self.config.batch_size == 0:
                self._update_policy(batch_log_probs, batch_rewards)
                batch_log_probs.clear()
                batch_rewards.clear()

            if episode % 50 == 0:
                avg_return = np.mean(returns[-50:])
                print(f"[REINFORCE] Episode {episode}/{self.config.episodes} | Avg Return (50 ep): {avg_return:.3f}")

        if batch_log_probs:
            self._update_policy(batch_log_probs, batch_rewards)

        return returns, entropy_per_episode

    def _discount_rewards(self, rewards: List[float]) -> List[float]:
        discounted = []
        running_return = 0.0
        for reward in reversed(rewards):
            running_return = reward + self.config.gamma * running_return
            discounted.insert(0, running_return)
        discounted = np.array(discounted)
        discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
        return discounted.tolist()

    def _update_policy(self, log_probs_batch: List[torch.Tensor], rewards_batch: List[torch.Tensor]) -> None:
        loss = 0.0
        for log_probs, rewards in zip(log_probs_batch, rewards_batch):
            loss += -(log_probs.to(self.device) * rewards).sum()

        loss /= len(log_probs_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



