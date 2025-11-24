"""
Training script for multiple RL algorithms on the Food Redistribution environment.

Algorithms:
- DQN
- PPO
- A2C
- Custom REINFORCE implementation (see reinforce.py)

Runs hyperparameter sweeps with exactly 10 combinations per algorithm.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.food_env import FoodRedistributionEnv
from reinforce import ReinforceAgent, ReinforceConfig

# Training budgets
SB3_TOTAL_TIMESTEPS = 50_000
REINFORCE_EPISODES = 2_000

# Store all run summaries for consolidated reporting
all_results: Dict[str, List[Dict]] = {"dqn": [], "ppo": [], "a2c": [], "reinforce": []}
LOGS_DIR: Path | None = None


def log_run_result(algo: str, run_id: str, mean_reward: float, std_reward: float, hyperparams: Dict) -> Path:
    """Persist per-run results and keep them for combined reporting."""
    summary = {
        "algo": algo,
        "run_id": run_id,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "hyperparams": hyperparams,
    }
    target_dir = LOGS_DIR or Path("logs")
    summary_path = target_dir / f"{run_id}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    all_results.setdefault(algo, []).append(summary)
    return summary_path

def make_env(seed: int | None = None):
    def _init():
        env = FoodRedistributionEnv()
        env.reset(seed=seed)
        return env

    return _init


def get_dqn_hyperparams() -> List[Dict]:
    """
    Generate exactly 10 hyperparameter combinations for DQN.
    
    Parameters varied:
    - Learning Rate: [1e-4, 3e-4, 1e-3]
    - Replay Buffer Size: [10_000, 50_000]
    - Batch Size: [32, 64, 128]
    - Gamma (discount factor): [0.95, 0.99]
    - Exploration Fraction: [0.1, 0.2]
    """
    learning_rates = [1e-4, 3e-4, 1e-3]
    buffer_sizes = [10_000, 50_000]
    batch_sizes = [32, 64, 128]
    gammas = [0.95, 0.99]
    exploration_fractions = [0.1, 0.2]
    
    # Generate combinations
    combinations = list(itertools.product(
        learning_rates, buffer_sizes, batch_sizes, gammas, exploration_fractions
    ))
    
    # Take exactly 10 combinations
    selected = combinations[:10]
    
    hyperparams = []
    for lr, buffer_size, batch_size, gamma, exploration_fraction in selected:
        hyperparams.append({
            "learning_rate": lr,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "exploration_fraction": exploration_fraction,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
        })
    
    return hyperparams


def get_ppo_hyperparams() -> List[Dict]:
    """
    Generate exactly 10 hyperparameter combinations for PPO.
    
    Parameters varied:
    - Learning Rate: [1e-4, 3e-4, 1e-3]
    - N Steps (rollout length): [128, 256, 512]
    - Batch Size: [32, 64, 128]
    - Gamma (discount factor): [0.95, 0.99]
    - N Epochs (optimization epochs): [4, 10]
    """
    learning_rates = [1e-4, 3e-4, 1e-3]
    n_steps = [128, 256, 512]
    batch_sizes = [32, 64, 128]
    gammas = [0.95, 0.99]
    n_epochs = [4, 10]
    
    # Generate combinations
    combinations = list(itertools.product(
        learning_rates, n_steps, batch_sizes, gammas, n_epochs
    ))
    
    # Take exactly 10 combinations
    selected = combinations[:10]
    
    hyperparams = []
    for lr, n_steps_val, batch_size, gamma, n_epochs_val in selected:
        hyperparams.append({
            "learning_rate": lr,
            "n_steps": n_steps_val,
            "batch_size": batch_size,
            "gamma": gamma,
            "n_epochs": n_epochs_val,
        })
    
    return hyperparams


def get_a2c_hyperparams() -> List[Dict]:
    """
    Generate exactly 10 hyperparameter combinations for A2C.
    
    Parameters varied:
    - Learning Rate: [1e-4, 3e-4, 1e-3, 3e-3]
    - N Steps (rollout length): [5, 10, 20]
    - Gamma (discount factor): [0.95, 0.99]
    - Entropy Coefficient: [0.0, 0.01]
    """
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
    n_steps = [5, 10, 20]
    gammas = [0.95, 0.99]
    ent_coefs = [0.0, 0.01]
    
    # Generate combinations
    combinations = list(itertools.product(
        learning_rates, n_steps, gammas, ent_coefs
    ))
    
    # Take exactly 10 combinations
    selected = combinations[:10]
    
    hyperparams = []
    for lr, n_steps_val, gamma, ent_coef in selected:
        hyperparams.append({
            "learning_rate": lr,
            "n_steps": n_steps_val,
            "gamma": gamma,
            "ent_coef": ent_coef,
        })
    
    return hyperparams


def get_reinforce_hyperparams() -> List[Dict]:
    """
    Generate exactly 10 hyperparameter combinations for REINFORCE.
    
    Parameters varied:
    - Learning Rate: [1e-4, 3e-4, 1e-3, 3e-3]
    - Gamma (discount factor): [0.95, 0.99]
    - Hidden Dimension (network size): [64, 128, 256]
    - Batch Size: [5, 10]
    """
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
    gammas = [0.95, 0.99]
    hidden_dims = [64, 128, 256]
    batch_sizes = [5, 10]
    
    # Generate combinations
    combinations = list(itertools.product(
        learning_rates, gammas, hidden_dims, batch_sizes
    ))
    
    # Take exactly 10 combinations
    selected = combinations[:10]
    
    hyperparams = []
    for lr, gamma, hidden_dim, batch_size in selected:
        hyperparams.append({
            "lr": lr,
            "gamma": gamma,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
        })
    
    return hyperparams


def train_sb3_models(logs_dir: Path, models_dir: Path) -> None:
    """Train SB3 models with hyperparameter sweeps."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    algorithms = {
        "dqn": (DQN, get_dqn_hyperparams),
        "ppo": (PPO, get_ppo_hyperparams),
        "a2c": (A2C, get_a2c_hyperparams),
    }
    
    for algo_name, (algo_class, get_hyperparams_fn) in algorithms.items():
        print(f"\n{'='*60}")
        print(f"Training {algo_name.upper()} with hyperparameter sweep")
        print(f"{'='*60}")
        
        hyperparams_list = get_hyperparams_fn()
        print(f"Running {len(hyperparams_list)} hyperparameter combinations...")
        
        for run_idx, hyperparams in enumerate(hyperparams_list, 1):
            print(f"\n--- {algo_name.upper()} Run {run_idx}/{len(hyperparams_list)} ---")
            print(f"Hyperparameters: {hyperparams}")
            
            # Create monitored environment for logging
            log_file = logs_dir / f"{algo_name}_run{run_idx:02d}_monitor.csv"
            
            def make_monitored_env():
                env = FoodRedistributionEnv()
                env = Monitor(env, filename=str(log_file))
                env.reset(seed=run_idx)
                return env
            
            vec_env = DummyVecEnv([make_monitored_env])
            
            # Create model with hyperparameters
            model = algo_class(
                "MlpPolicy",
                vec_env,
                verbose=0,  # Reduce verbosity for multiple runs
                tensorboard_log=str(logs_dir / "tensorboard" / algo_name),
                policy_kwargs={"net_arch": [128, 128]},
                **hyperparams
            )
            
            # Train
            model.learn(total_timesteps=SB3_TOTAL_TIMESTEPS)
            
            # Save with unique identifier
            run_id = f"{algo_name}_run{run_idx:02d}"
            save_path = models_dir / run_id
            model.save(save_path)
            print(f"Saved {algo_name.upper()} model to {save_path}.zip")

            vec_env.close()
            
            # Read monitor CSV and compute summary statistics
            try:
                # Monitor CSV has an initial metadata line starting with '#'
                # followed by a header row (r,l,t). We skip comment lines
                # and parse the rest with csv.DictReader.
                rewards: List[float] = []
                with open(log_file, "r", newline="") as f:
                    def _valid_rows():
                        for line in f:
                            # Skip empty lines and metadata comments
                            if not line.strip() or line.startswith("#"):
                                continue
                            yield line

                    reader = csv.DictReader(_valid_rows())
                    for row in reader:
                        reward_str = row.get("r")
                        if reward_str is None or reward_str == "":
                            continue
                        rewards.append(float(reward_str))
                
                if rewards:
                    mean_reward = float(np.mean(rewards))
                    std_reward = float(np.std(rewards))
                else:
                    mean_reward = 0.0
                    std_reward = 0.0
                
                summary_path = log_run_result(
                    algo=algo_name,
                    run_id=run_id,
                    mean_reward=mean_reward,
                    std_reward=std_reward,
                    hyperparams=hyperparams,
                )
                print(f"Saved summary to {summary_path}")
                print(f"  Mean Reward: {mean_reward:.4f}, Std Reward: {std_reward:.4f}")
            except Exception as e:
                print(f"Warning: Could not generate summary for {run_id}: {e}")


def train_reinforce(logs_dir: Path, models_dir: Path) -> None:
    """Train REINFORCE with hyperparameter sweeps."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    hyperparams_list = get_reinforce_hyperparams()
    print(f"\n{'='*60}")
    print(f"Training REINFORCE with hyperparameter sweep")
    print(f"{'='*60}")
    print(f"Running {len(hyperparams_list)} hyperparameter combinations...")
    
    for run_idx, hyperparams in enumerate(hyperparams_list, 1):
        print(f"\n--- REINFORCE Run {run_idx}/{len(hyperparams_list)} ---")
        print(f"Hyperparameters: {hyperparams}")
        
        env = FoodRedistributionEnv()
        config = ReinforceConfig(
            episodes=REINFORCE_EPISODES,
            gamma=hyperparams["gamma"],
            lr=hyperparams["lr"],
            hidden_dim=hyperparams["hidden_dim"],
            batch_size=hyperparams["batch_size"],
        )
        
        agent = ReinforceAgent(env.observation_space.shape[0], env.action_space.n, config)
        returns, entropy_per_episode = agent.train(env)
        
        run_id = f"reinforce_run{run_idx:02d}"
        
        # Save policy weights to models_dir
        policy_path = models_dir / f"{run_id}_policy.pt"
        torch.save(agent.policy.state_dict(), policy_path)
        print(f"Saved REINFORCE policy to {policy_path}")
        
        # Save logs (returns and entropy) to logs_dir
        returns_path = logs_dir / f"{run_id}_returns.csv"
        entropy_path = logs_dir / f"{run_id}_entropy.csv"
        
        np.savetxt(returns_path, np.array(returns), delimiter=",")
        np.savetxt(entropy_path, np.array(entropy_per_episode), delimiter=",")
        
        print(f"Saved REINFORCE returns to {returns_path}")
        print(f"Saved REINFORCE entropy to {entropy_path}")
        
        # Compute summary statistics from returns
        mean_reward = float(np.mean(returns))
        std_reward = float(np.std(returns))
        
        summary_path = log_run_result(
            algo="reinforce",
            run_id=run_id,
            mean_reward=mean_reward,
            std_reward=std_reward,
            hyperparams=hyperparams,
        )
        
        print(f"Saved summary to {summary_path}")
        print(f"  Mean Reward: {mean_reward:.4f}, Std Reward: {std_reward:.4f}")


def summarize_all_runs(logs_dir: Path) -> None:
    """Aggregate per-run summaries and print overview tables."""
    summary_files = sorted(logs_dir.glob("*_summary.json"))
    if not summary_files:
        print(f"\nNo summary files found in {logs_dir}.")
        return

    grouped: Dict[str, List[Dict]] = {}
    for file_path in summary_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not load summary {file_path}: {exc}")
            continue

        algo_name = data.get("algo", "unknown")
        grouped.setdefault(algo_name, []).append(data)

    print("\n================ Overall Training Summary ================")
    for algo_name in sorted(grouped.keys()):
        runs = sorted(grouped[algo_name], key=lambda x: x.get("run_id", ""))
        print(f"\n--- {algo_name.upper()} ({len(runs)} runs) ---")
        print(f"{'Run ID':<18}{'Mean':>12}{'Std':>12}")
        print("-" * 42)
        for run in runs:
            mean_val = run.get("mean_reward", float("nan"))
            std_val = run.get("std_reward", float("nan"))
            run_id = run.get("run_id", "unknown")
            print(f"{run_id:<18}{mean_val:>12.4f}{std_val:>12.4f}")

        best_run = max(runs, key=lambda x: x.get("mean_reward", float("-inf")))
        print("\nBest run:")
        print(f"  Run ID       : {best_run.get('run_id', 'unknown')}")
        print(f"  Mean Reward  : {best_run.get('mean_reward', float('nan')):.4f}")
        print(f"  Std Reward   : {best_run.get('std_reward', float('nan')):.4f}")
        print("  Hyperparams  :")
        hyperparams = best_run.get("hyperparams", {})
        for key, value in hyperparams.items():
            print(f"    - {key}: {value}")
        print("-" * 42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agents for food redistribution.")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory to store training logs (CSVs, tensorboard).")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to store model weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    models_dir = Path(args.models_dir)
    global LOGS_DIR
    LOGS_DIR = logs_dir
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    train_sb3_models(logs_dir=logs_dir, models_dir=models_dir)
    train_reinforce(logs_dir=logs_dir, models_dir=models_dir)
    summarize_all_runs(logs_dir=logs_dir)

    best_per_algorithm: Dict[str, Dict] = {}
    for algo, runs in all_results.items():
        if not runs:
            continue
        best_per_algorithm[algo] = max(runs, key=lambda r: r.get("mean_reward", float("-inf")))

    combined = {
        "all_runs": all_results,
        "best_per_algorithm": best_per_algorithm,
    }
    combined_path = logs_dir / "all_algorithms_summary.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n✔ Saved combined summary to {combined_path}")


if __name__ == "__main__":
    main()



