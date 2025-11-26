import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.food_env import FoodRedistributionEnv

def make_env():
    return FoodRedistributionEnv(render_mode="human")

def main():
    vec_env = DummyVecEnv([make_env])
    model = PPO.load("models/best models/overallbest_ppo", env=vec_env)
    n_episodes = 20  # Increased from 5 to run longer

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        print(f"\n==================== Episode {ep+1} ====================")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += reward[0]
            metrics = info[0] if isinstance(info, list) else info
            waste = metrics.get("waste", 0)
            unmet = metrics.get("unmet_demand", 0)
            fairness = metrics.get("fairness", 0)
            delivered_ratio = metrics.get("delivered_ratio", 0)

            print(
                f"Step {step:02d} | action={int(action[0])} | "
                f"reward={reward[0]:.3f} | waste={waste:.2f} | "
                f"unmet={unmet:.2f} | fairness={fairness:.2f} | "
                f"delivery_eff={delivered_ratio:.2f}"
            )

            try:
                vec_env.envs[0].render()
                time.sleep(0.2)
            except Exception:
                pass

            step += 1

        print(f"Total episode reward: {ep_reward:.3f}")

    vec_env.close()

if __name__ == "__main__":
    main()

