from __future__ import annotations

import time
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.food_env import FoodRedistributionEnv
from visualization_scripts.food_dashboard import FoodDashboard


MODEL_PATH = Path("trained_models") / "best models" / "overallbest_ppo.zip"
N_EPISODES = 100
STEP_DELAY = 0.15


def make_env():
    # Use rgb_array so the dashboard receives ModernGL frames.
    return FoodRedistributionEnv(render_mode="rgb_array")


def main() -> None:
    vec_env = DummyVecEnv([make_env])
    raw_env = vec_env.envs[0]
    model = PPO.load(str(MODEL_PATH), device="cpu")
    model.set_env(vec_env)

    dashboard = FoodDashboard(width=1200, height=750)

    try:
        for episode in range(N_EPISODES):
            obs = vec_env.reset()
            raw_env = vec_env.envs[0]
            done = np.array([False])
            episode_reward = 0.0
            step = 0

            print(f"\n==================== Episode {episode + 1} ====================")

            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                episode_reward += reward[0]
                metrics = info[0] if isinstance(info, (list, tuple)) else info

                print(
                    f"Step {step:02d} | action={int(action[0])} | "
                    f"reward={reward[0]:.3f} | waste={metrics.get('waste', 0):.2f} | "
                    f"unmet={metrics.get('unmet_demand', 0):.2f} | "
                    f"fairness={metrics.get('fairness', 0):.3f}"
                )

                if not dashboard.render_frame(raw_env, episode + 1, N_EPISODES, episode_reward):
                    print("Dashboard window closed. Exiting simulation.")
                    return

                time.sleep(STEP_DELAY)
                step += 1

            print(f"Episode {episode + 1} finished with reward {episode_reward:.3f}")

    finally:
        dashboard.cleanup()
        vec_env.close()


if __name__ == "__main__":
    main()

