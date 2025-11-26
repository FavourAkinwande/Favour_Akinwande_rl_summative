"""
Comprehensive evaluation script for all four RL algorithms:
DQN, PPO, A2C, and REINFORCE.

Evaluates each model over 30 episodes and produces detailed metrics,
comparison tables, and analysis summaries.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.food_env import FoodRedistributionEnv
from training_scripts.reinforce import ReinforceAgent, ReinforceConfig, PolicyNet

# Model paths (flexible - checks multiple locations)
MODEL_PATHS = {
    "dqn": [
        Path("models/best models/best_dqn.zip"),
        Path("models/best models/dqn_run04.zip"),
        Path("models/best_dqn.zip"),
        Path("models/dqn_run04.zip"),
    ],
    "ppo": [
        Path("models/best models/best_ppo.zip"),
        Path("models/best models/overallbest_ppo.zip"),
        Path("models/best models/ppo_run04.zip"),
        Path("models/best_ppo.zip"),
        Path("models/ppo_run04.zip"),
    ],
    "a2c": [
        Path("models/best models/best_a2c.zip"),
        Path("models/best models/a2c_run01.zip"),
        Path("models/best_a2c.zip"),
        Path("models/a2c_run01.zip"),
    ],
    "reinforce": [
        Path("models/best models/best_reinforce.zip"),
        Path("models/best models/reinforce_run07_policy.pt"),
        Path("models/best_reinforce.zip"),
        Path("models/reinforce_run07_policy.pt"),
    ],
}

N_EVAL_EPISODES = 30


def find_model_path(algo: str) -> Optional[Path]:
    """Find the best model path for an algorithm."""
    for path in MODEL_PATHS[algo]:
        if path.exists():
            return path
    return None


def make_env(seed: Optional[int] = None):
    """Create environment factory for DummyVecEnv."""
    def _init():
        env = FoodRedistributionEnv()
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def evaluate_sb3_model(model, env, n_episodes: int = N_EVAL_EPISODES) -> Dict[str, float]:
    """
    Evaluate a Stable-Baselines3 model (DQN, PPO, A2C).
    
    Returns dictionary with metrics:
    - mean_reward, std_reward, max_reward, min_reward
    - mean_episode_length, stability_score, success_rate
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    
    for episode in range(n_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        done = np.array([False])
        episode_reward = 0.0
        episode_length = 0
        
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            # DummyVecEnv returns (obs, reward, done, info) - 4 values
            if len(step_result) == 4:
                obs, reward, done, _ = step_result
            elif len(step_result) == 5:
                # Gymnasium API (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, _ = step_result
                done = terminated | truncated
            else:
                raise ValueError(f"Unexpected step result length: {len(step_result)}")
            episode_reward += float(reward[0] if isinstance(reward, np.ndarray) else reward)
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    rewards_array = np.array(episode_rewards)
    lengths_array = np.array(episode_lengths)
    
    return {
        "mean_reward": float(np.mean(rewards_array)),
        "std_reward": float(np.std(rewards_array)),
        "max_reward": float(np.max(rewards_array)),
        "min_reward": float(np.min(rewards_array)),
        "mean_episode_length": float(np.mean(lengths_array)),
        "stability_score": float(np.var(rewards_array)),  # Variance as stability measure
        "success_rate": float(np.sum(rewards_array > 0) / len(rewards_array)),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def evaluate_reinforce_model(agent: ReinforceAgent, env, n_episodes: int = N_EVAL_EPISODES) -> Dict[str, float]:
    """
    Evaluate a REINFORCE agent.
    
    Returns dictionary with metrics (same format as SB3 models).
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    
    agent.policy.eval()  # Set to evaluation mode
    
    for episode in range(n_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            obs_tensor = torch.from_numpy(obs).float().to(agent.device)
            with torch.no_grad():
                action_probs = agent.policy(obs_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()  # Deterministic would be argmax, but we use sample for consistency
            
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            episode_reward += float(reward)
            episode_length += 1
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    rewards_array = np.array(episode_rewards)
    lengths_array = np.array(episode_lengths)
    
    return {
        "mean_reward": float(np.mean(rewards_array)),
        "std_reward": float(np.std(rewards_array)),
        "max_reward": float(np.max(rewards_array)),
        "min_reward": float(np.min(rewards_array)),
        "mean_episode_length": float(np.mean(lengths_array)),
        "stability_score": float(np.var(rewards_array)),
        "success_rate": float(np.sum(rewards_array > 0) / len(rewards_array)),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def load_reinforce_model(model_path: Path, env) -> ReinforceAgent:
    """Load a REINFORCE model from a .pt file."""
    # Load state dict first to infer hidden_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    
    # Infer hidden_dim from the first layer weight shape
    # First layer: net.0.weight has shape [hidden_dim, obs_dim]
    first_layer_weight = state_dict.get("net.0.weight", None)
    if first_layer_weight is not None:
        hidden_dim = int(first_layer_weight.shape[0])
    else:
        # Fallback: try common values
        hidden_dim = 64
    
    config = ReinforceConfig(
        episodes=2000,  # Not used in eval, but needed for agent init
        gamma=0.99,
        lr=1e-3,
        hidden_dim=hidden_dim,
        batch_size=5,
    )
    
    agent = ReinforceAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config,
    )
    
    # Load the policy state dict
    agent.policy.load_state_dict(state_dict)
    agent.policy.eval()
    
    return agent


def evaluate_model(model_or_agent, env, algo: str) -> Dict[str, float]:
    """
    Unified evaluation function that handles both SB3 models and REINFORCE.
    """
    if algo == "reinforce":
        return evaluate_reinforce_model(model_or_agent, env, N_EVAL_EPISODES)
    else:
        return evaluate_sb3_model(model_or_agent, env, N_EVAL_EPISODES)


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 100)
    print(f"{'Algorithm':<12} | {'Mean':<10} | {'Std':<10} | {'Max':<10} | {'Min':<10} | {'Stability':<12} | {'Success Rate':<12}")
    print("-" * 100)
    
    for algo in ["dqn", "ppo", "a2c", "reinforce"]:
        if algo not in results:
            continue
        r = results[algo]
        print(
            f"{algo.upper():<12} | "
            f"{r['mean_reward']:>10.4f} | "
            f"{r['std_reward']:>10.4f} | "
            f"{r['max_reward']:>10.4f} | "
            f"{r['min_reward']:>10.4f} | "
            f"{r['stability_score']:>12.4f} | "
            f"{r['success_rate']:>12.2%}"
        )
    print("=" * 100)


def save_results_csv(results: Dict[str, Dict[str, float]], output_path: Path = Path("evaluation_results.csv")) -> None:
    """Save evaluation results to CSV."""
    fieldnames = [
        "algorithm", "mean_reward", "std_reward", "max_reward", "min_reward",
        "mean_episode_length", "stability_score", "success_rate"
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for algo, metrics in results.items():
            row = {"algorithm": algo}
            for key in fieldnames[1:]:
                row[key] = metrics[key]
            writer.writerow(row)
    
    print(f"\n✔ Saved evaluation results to {output_path}")


def print_analysis_summary(results: Dict[str, Dict[str, float]]) -> None:
    """Print detailed analysis summary."""
    print("\n" + "=" * 100)
    print("ANALYSIS SUMMARY")
    print("=" * 100)
    
    # Find best model
    best_algo = max(results.keys(), key=lambda k: results[k]["mean_reward"])
    best_mean = results[best_algo]["mean_reward"]
    
    # Find most stable (lowest variance)
    most_stable = min(results.keys(), key=lambda k: results[k]["stability_score"])
    
    # Find least stable (highest variance)
    least_stable = max(results.keys(), key=lambda k: results[k]["stability_score"])
    
    print(f"\n🏆 BEST PERFORMING MODEL: {best_algo.upper()}")
    print(f"   Mean Reward: {best_mean:.4f}")
    print(f"   This model achieved the highest average reward over {N_EVAL_EPISODES} episodes.")
    
    # Why it likely performed best
    print(f"\n📊 WHY {best_algo.upper()} LIKELY PERFORMED BEST:")
    best_metrics = results[best_algo]
    reasons = []
    if best_metrics["success_rate"] > 0.8:
        reasons.append("High success rate (>80%)")
    if best_metrics["std_reward"] < np.mean([r["std_reward"] for r in results.values()]):
        reasons.append("Lower variance (more consistent)")
    if best_metrics["max_reward"] > np.mean([r["max_reward"] for r in results.values()]):
        reasons.append("Achieved higher peak rewards")
    if not reasons:
        reasons.append("Superior average performance across all metrics")
    for reason in reasons:
        print(f"   • {reason}")
    
    print(f"\n⚖️  STABILITY ANALYSIS:")
    print(f"   Most Stable: {most_stable.upper()} (variance: {results[most_stable]['stability_score']:.4f})")
    print(f"   Least Stable: {least_stable.upper()} (variance: {results[least_stable]['stability_score']:.4f})")
    
    print(f"\n🔍 ALGORITHM-SPECIFIC ANALYSIS:")
    for algo in ["dqn", "ppo", "a2c", "reinforce"]:
        if algo not in results:
            continue
        r = results[algo]
        print(f"\n{algo.upper()}:")
        print(f"   Mean Reward: {r['mean_reward']:.4f} ± {r['std_reward']:.4f}")
        print(f"   Success Rate: {r['success_rate']:.1%}")
        print(f"   Stability: {r['stability_score']:.4f} variance")
        
        # Strengths/weaknesses
        strengths = []
        weaknesses = []
        
        if r["mean_reward"] > np.mean([v["mean_reward"] for v in results.values()]):
            strengths.append("Above-average mean reward")
        else:
            weaknesses.append("Below-average mean reward")
        
        if r["stability_score"] < np.mean([v["stability_score"] for v in results.values()]):
            strengths.append("More stable (lower variance)")
        else:
            weaknesses.append("Less stable (higher variance)")
        
        if r["success_rate"] > 0.7:
            strengths.append("High success rate")
        elif r["success_rate"] < 0.5:
            weaknesses.append("Low success rate")
        
        if strengths:
            print(f"   Strengths: {', '.join(strengths)}")
        if weaknesses:
            print(f"   Weaknesses: {', '.join(weaknesses)}")
    
    print("\n" + "=" * 100)


def main() -> None:
    """Main evaluation function."""
    print("=" * 100)
    print("RL ALGORITHM EVALUATION")
    print("=" * 100)
    print(f"Evaluating all models over {N_EVAL_EPISODES} episodes each...")
    print()
    
    results: Dict[str, Dict[str, float]] = {}
    
    # Evaluate each algorithm
    for algo in ["dqn", "ppo", "a2c", "reinforce"]:
        print(f"\n{'='*60}")
        print(f"Evaluating {algo.upper()}...")
        print(f"{'='*60}")
        
        model_path = find_model_path(algo)
        if model_path is None:
            print(f"⚠️  Warning: Could not find model for {algo}. Skipping...")
            continue
        
        print(f"Loading model from: {model_path}")
        
        # Create environment
        vec_env = DummyVecEnv([make_env()])
        raw_env = vec_env.envs[0]
        
        try:
            # Load model
            if algo == "reinforce":
                if model_path.suffix == ".pt":
                    agent = load_reinforce_model(model_path, raw_env)
                    model_or_agent = agent
                else:
                    print(f"⚠️  REINFORCE model should be .pt file, found {model_path.suffix}")
                    vec_env.close()
                    continue
            else:
                # SB3 models
                if algo == "dqn":
                    model = DQN.load(str(model_path), env=vec_env)
                elif algo == "ppo":
                    model = PPO.load(str(model_path), env=vec_env)
                elif algo == "a2c":
                    model = A2C.load(str(model_path), env=vec_env)
                else:
                    vec_env.close()
                    continue
                model_or_agent = model
            
            # Evaluate
            print(f"Running {N_EVAL_EPISODES} evaluation episodes...")
            # For REINFORCE, use raw_env; for SB3, use vec_env
            eval_env = raw_env if algo == "reinforce" else vec_env
            metrics = evaluate_model(model_or_agent, eval_env, algo)
            results[algo] = metrics
            
            print(f"✓ {algo.upper()} evaluation complete:")
            print(f"  Mean Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
            print(f"  Success Rate: {metrics['success_rate']:.1%}")
            
            vec_env.close()
            
        except Exception as e:
            print(f"❌ Error evaluating {algo}: {e}")
            import traceback
            traceback.print_exc()
            vec_env.close()
            continue
    
    # Print comparison table
    if results:
        print_comparison_table(results)
        
        # Save to CSV
        save_results_csv(results)
        
        # Print analysis
        print_analysis_summary(results)
    else:
        print("\n❌ No models were successfully evaluated.")


if __name__ == "__main__":
    main()

