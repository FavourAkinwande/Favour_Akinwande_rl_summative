"""
Interactive pygame visualization for the Food Redistribution environment.
"""

from __future__ import annotations

import argparse
import time

import pygame

from environment.food_env import FoodRedistributionEnv


def run_viewer(agent: str = "random", fps: int = 2) -> None:
    env = FoodRedistributionEnv(render_mode="human")
    obs, _ = env.reset()

    running = True
    last_action = None

    try:
        while running:
            if agent == "random":
                action = env.action_space.sample()
            else:
                raise ValueError(f"Unsupported agent type: {agent}")

            obs, reward, terminated, truncated, info = env.step(action)
            last_action = (action, reward, info)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if terminated or truncated:
                obs, _ = env.reset()

            time.sleep(1 / fps)
    finally:
        env.close()
        if last_action:
            action, reward, info = last_action
            print(f"Last action: {action} | Reward: {reward:.3f} | Info: {info}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pygame viewer for the Food Redistribution environment.")
    parser.add_argument("--agent", type=str, default="random", help="Agent type (currently only 'random').")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for visualization.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_viewer(agent=args.agent, fps=args.fps)



