"""
Train and evaluate the best PPO configuration discovered during hyperparameter sweeps.

Usage:
    python train_best_ppo.py
"""

from __future__ import annotations

import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.food_env import FoodRedistributionEnv


TOTAL_TIMESTEPS = 500_000
N_EVAL_EPISODES = 100
MODEL_PATH = Path("models") / "ppo_best_500k"

BEST_PPO_PARAMS = dict(
    learning_rate=0.0001,
    n_steps=128,
    batch_size=32,
    gamma=0.99,
    n_epochs=10,
    verbose=1,
)

# Set device="cuda" on Kaggle GPU, "cpu" locally if needed.
DEVICE = "cpu"


def make_env():
    """Return a new instance of the Food Redistribution environment."""
    return FoodRedistributionEnv()


def build_vec_env():
    """Wrap the environment in a DummyVecEnv for SB3 compatibility."""
    return DummyVecEnv([make_env])


def train_or_load(model_path: Path, vec_env: DummyVecEnv) -> PPO:
    """
    Train the PPO model for TOTAL_TIMESTEPS, or load it if the file already exists.
    """
    if model_path.with_suffix(".zip").exists():
        print(f"Loading existing PPO model from {model_path}.zip")
        return PPO.load(model_path, env=vec_env, device=DEVICE)

    print(f"Training PPO with best hyperparameters for {TOTAL_TIMESTEPS:,} timesteps...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        device=DEVICE,
        **BEST_PPO_PARAMS,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    os.makedirs(model_path.parent, exist_ok=True)
    model.save(model_path)
    print(f"Saved PPO model to {model_path}.zip")
    return model


def main() -> None:
    vec_env = build_vec_env()
    model = train_or_load(MODEL_PATH, vec_env)

    print(f"Evaluating PPO over {N_EVAL_EPISODES} episodes...")
    mean_reward, std_reward = evaluate_policy(
        model,
        vec_env,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )
    print(
        f"Final PPO (500k) -> Mean reward: {mean_reward:.4f} | "
        f"Std: {std_reward:.4f} over {N_EVAL_EPISODES} episodes"
    )


if __name__ == "__main__":
    main()

