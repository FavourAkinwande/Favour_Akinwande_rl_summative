"""
Custom reinforcement learning environment for surplus food redistribution.

The environment simulates three retailers and three communities that must be
connected by a single truck choosing (retailer -> community) routes across a
24-step day. It tracks food freshness, supply, demand, fairness, transport cost,
and waste, rewarding the agent for efficient and equitable deliveries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pygame
from gymnasium import Env, spaces


@dataclass
class FoodEnvConfig:
    """Configuration values for the food redistribution environment."""

    num_retailers: int = 3
    num_communities: int = 3
    day_length: int = 24
    truck_capacity: float = 40.0
    max_supply: float = 120.0
    max_demand: float = 100.0
    freshness_decay: float = 0.04
    supply_decay: float = 4.0
    demand_fluctuation: float = 5.0
    waste_penalty_weight: float = 0.02
    fairness_penalty_weight: float = 0.2
    transport_penalty_weight: float = 0.1
    idle_penalty: float = 0.1
    distance_matrix: Optional[np.ndarray] = None


class FoodRedistributionEnv(Env):
    """
    Simulation of surplus food redistribution in an African urban city.

    Observation vector (all normalized):
    [time (1), supply (3), freshness (3), demand (3)] -> 10 dims.
    Action space: 9 discrete (retailer * community).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config: Optional[FoodEnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or FoodEnvConfig()
        self.render_mode = render_mode

        self.num_retailers = self.config.num_retailers
        self.num_communities = self.config.num_communities
        self.day_length = self.config.day_length

        self.action_space = spaces.Discrete(self.num_retailers * self.num_communities)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1 + self.num_retailers * 2 + self.num_communities,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self._distance_matrix = (
            self.config.distance_matrix
            if self.config.distance_matrix is not None
            else self._default_distance_matrix()
        )
        self._max_distance = np.max(self._distance_matrix)

        # Simulation state
        self.time_step = 0
        self.supplies: np.ndarray
        self.freshness: np.ndarray
        self.demands: np.ndarray
        self.initial_demands: np.ndarray
        self.delivered_totals: np.ndarray

        # Rendering state
        self._window: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._canvas: Optional[pygame.Surface] = None
        self.window_size = (900, 600)
        self.retailer_positions = [(120, 120), (120, 300), (120, 480)]
        self.community_positions = [(780, 120), (780, 300), (780, 480)]

    # Gym API -----------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.time_step = 0
        self.supplies = self._rng.uniform(0.4, 0.9, size=self.num_retailers) * self.config.max_supply
        self.freshness = self._rng.uniform(0.6, 1.0, size=self.num_retailers)
        self.demands = self._rng.uniform(0.4, 0.85, size=self.num_communities) * self.config.max_demand
        self.initial_demands = self.demands.copy()
        self.delivered_totals = np.zeros(self.num_communities, dtype=float)

        observation = self._get_observation()
        info = self._info()
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        retailer_idx, community_idx = divmod(action, self.num_communities)

        supply = self.supplies[retailer_idx]
        demand = self.demands[community_idx]
        deliverable = min(self.config.truck_capacity, supply, demand)

        freshness = self.freshness[retailer_idx]
        travel_cost = self._distance_matrix[retailer_idx, community_idx] / self._max_distance

        delivery_reward = deliverable / self.config.truck_capacity
        freshness_reward = (deliverable / self.config.truck_capacity) * freshness * 0.5
        idle_penalty = self.config.idle_penalty if deliverable == 0 else 0.0

        self.supplies[retailer_idx] -= deliverable
        self.demands[community_idx] -= deliverable
        self.delivered_totals[community_idx] += deliverable

        # Spoilage & dynamics
        self.supplies = np.clip(self.supplies - self.config.supply_decay, 0, self.config.max_supply)
        self.freshness = np.clip(self.freshness - self.config.freshness_decay, 0, 1)
        demand_noise = self._rng.normal(0, self.config.demand_fluctuation, size=self.num_communities)
        self.demands = np.clip(self.demands + demand_noise, 0, self.config.max_demand)

        waste_penalty = self.config.waste_penalty_weight * (np.sum(self.supplies) / (self.num_retailers * self.config.max_supply))
        fairness_penalty = self.config.fairness_penalty_weight * self._fairness_std()
        transport_penalty = self.config.transport_penalty_weight * travel_cost

        reward = delivery_reward + freshness_reward - waste_penalty - fairness_penalty - transport_penalty - idle_penalty

        self.time_step += 1
        terminated = self.time_step >= self.day_length
        truncated = False

        observation = self._get_observation()
        info = self._info(delivered=deliverable, retailer=retailer_idx, community=community_idx)

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode {self.render_mode}.")

        if self._canvas is None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human" and self._window is None:
                self._window = pygame.display.set_mode(self.window_size)
                pygame.display.set_caption("Food Redistribution Environment")
            self._canvas = pygame.Surface(self.window_size)
            self._font = pygame.font.SysFont("arial", 18)
            self._clock = pygame.time.Clock()

        self._canvas.fill((25, 29, 35))
        self._draw_nodes()
        self._draw_status_panel()

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            assert self._window is not None
            self._window.blit(self._canvas, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None

        frame = pygame.surfarray.array3d(self._canvas)
        return np.transpose(frame, (1, 0, 2))

    def close(self) -> None:
        if self._window is not None:
            pygame.display.quit()
            self._window = None
        if self._canvas is not None:
            pygame.quit()
            self._canvas = None

    # Helpers -----------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        time_feature = np.array([self.time_step / self.day_length], dtype=np.float32)
        supply_feature = (self.supplies / self.config.max_supply).astype(np.float32)
        freshness_feature = self.freshness.astype(np.float32)
        demand_feature = (self.demands / self.config.max_demand).astype(np.float32)
        return np.concatenate([time_feature, supply_feature, freshness_feature, demand_feature]).astype(np.float32)

    def _info(self, delivered: float = 0.0, retailer: int = -1, community: int = -1) -> Dict[str, Any]:
        return {
            "time_step": self.time_step,
            "delivered": delivered,
            "retailer": retailer,
            "community": community,
            "supplies": self.supplies.copy(),
            "demands": self.demands.copy(),
            "freshness": self.freshness.copy(),
            "fairness_std": self._fairness_std(),
        }

    def _fairness_std(self) -> float:
        demand_totals = np.where(self.initial_demands > 0, self.initial_demands, 1.0)
        ratios = self.delivered_totals / demand_totals
        return float(np.std(ratios))

    def _default_distance_matrix(self) -> np.ndarray:
        base = np.array(
            [
                [3.0, 6.0, 9.0],
                [4.0, 2.0, 7.0],
                [5.0, 6.5, 3.0],
            ]
        )
        return base

    def _draw_nodes(self) -> None:
        assert self._canvas is not None
        for idx, (x, y) in enumerate(self.retailer_positions[: self.num_retailers]):
            supply = self.supplies[idx]
            freshness = self.freshness[idx]
            color = (50, 200, 120) if supply > 1 else (90, 90, 90)
            pygame.draw.circle(self._canvas, color, (x, y), 30)
            text = self._font.render(f"R{idx+1}", True, (0, 0, 0))
            self._canvas.blit(text, (x - 12, y - 10))
            info = self._font.render(f"S:{supply:.1f} F:{freshness:.2f}", True, (240, 240, 240))
            self._canvas.blit(info, (x - 40, y + 40))

        for idx, (x, y) in enumerate(self.community_positions[: self.num_communities]):
            demand = self.demands[idx]
            color = (220, 120, 70) if demand > 1 else (90, 90, 90)
            pygame.draw.circle(self._canvas, color, (x, y), 30)
            text = self._font.render(f"C{idx+1}", True, (0, 0, 0))
            self._canvas.blit(text, (x - 12, y - 10))
            info = self._font.render(f"D:{demand:.1f}", True, (240, 240, 240))
            self._canvas.blit(info, (x - 50, y + 40))

    def _draw_status_panel(self) -> None:
        assert self._canvas is not None
        panel_rect = pygame.Rect(330, 20, 240, 560)
        pygame.draw.rect(self._canvas, (38, 45, 55), panel_rect, border_radius=8)
        pygame.draw.rect(self._canvas, (80, 120, 200), panel_rect, width=2, border_radius=8)

        lines = [
            f"Time step: {self.time_step}/{self.day_length}",
            f"Total delivered: {np.sum(self.delivered_totals):.1f}",
            f"Fairness std: {self._fairness_std():.3f}",
            f"Avg freshness: {np.mean(self.freshness):.2f}",
            f"Remaining supply: {np.sum(self.supplies):.1f}",
            f"Remaining demand: {np.sum(self.demands):.1f}",
        ]
        for idx, line in enumerate(lines):
            text = self._font.render(line, True, (235, 235, 235))
            self._canvas.blit(text, (panel_rect.x + 12, panel_rect.y + 15 + idx * 24))


