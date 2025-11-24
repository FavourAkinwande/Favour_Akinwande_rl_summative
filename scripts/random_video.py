"""
Generate a video of a random agent interacting with the environment.
Shows a pygame window while recording.
"""

from __future__ import annotations

from pathlib import Path

import argparse
import imageio
import pygame
from tqdm import trange

from environment.food_env import FoodRedistributionEnv


def record_random_policy(steps: int = 60, output_path: str = "videos/random_agent.mp4") -> None:
    # Use rgb_array mode to capture frames, but we'll also show a window
    env = FoodRedistributionEnv(render_mode="rgb_array")
    frames = []
    obs, _ = env.reset()

    # Initialize pygame window for display
    pygame.init()
    window = pygame.display.set_mode(env.window_size)
    pygame.display.set_caption("Food Redistribution Environment - Recording")
    clock = pygame.time.Clock()

    for _ in trange(steps, desc="Recording random agent"):
        # Handle pygame events (e.g., window close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                return
        
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        
        if frame is not None:
            frames.append(frame)
            # Display the frame in the pygame window
            # Convert numpy array to pygame surface
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            window.blit(frame_surface, (0, 0))
            pygame.display.flip()
            clock.tick(env.metadata["render_fps"])
        
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    pygame.quit()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_file, frames, fps=env.metadata["render_fps"])
    print(f"Saved random agent video to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a random agent video.")
    parser.add_argument("--steps", type=int, default=60, help="Number of steps to record.")
    parser.add_argument("--output", type=str, default="videos/random_agent.mp4", help="Output MP4 path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_random_policy(steps=args.steps, output_path=args.output)


