import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stable_baselines3 import A2C, DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from environment.food_env import FoodRedistributionEnv
    from training_scripts.reinforce import ReinforceAgent, ReinforceConfig, PolicyNet
    import torch
    HAS_RL_IMPORTS = True
except ImportError:
    HAS_RL_IMPORTS = False
    print("Warning: RL libraries not available. Generalization analysis will be skipped.")

CSV_PATHS = [
    "evaluation_results.csv",
    os.path.join("model_evaluation", "evaluation_results.csv"),
]
OUTPUT_DIR = "figures"


def load_data(csv_paths) -> pd.DataFrame:
    for path in csv_paths:
        if os.path.exists(path):
            return pd.read_csv(path)

    search_paths = "\n".join(csv_paths)
    raise FileNotFoundError(
        "Could not locate evaluation_results.csv. Paths checked:\n" + search_paths
    )


def plot_mean_reward(df: pd.DataFrame) -> None:
    algorithms = df["algorithm"]
    mean_rewards = df["mean_reward"]
    std_rewards = df["std_reward"]

    plt.figure(figsize=(8, 5))
    plt.bar(
        algorithms,
        mean_rewards,
        yerr=std_rewards,
        capsize=8,
        color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"],
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Mean Reward with Standard Deviation", fontsize=14)
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Mean Reward", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mean_reward_errorbars.png"), dpi=300)
    plt.close()


def load_monitor_rewards(monitor_path: str) -> np.ndarray:
    """
    Load episode rewards from a Stable-Baselines3 Monitor CSV.
    Skips comment lines starting with '#', then uses column 'r'.
    """
    if not os.path.exists(monitor_path):
        raise FileNotFoundError(f"Monitor file not found: {monitor_path}")

    # Use pandas and tell it to ignore comment lines starting with '#'
    df = pd.read_csv(monitor_path, comment="#")

    if "r" in df.columns:
        rewards = df["r"].to_numpy(dtype=float)
    else:
        if df.shape[1] == 1:
            rewards = df.iloc[:, 0].to_numpy(dtype=float)
        else:
            raise ValueError(
                "Expected column 'r' or a single unnamed column in monitor file."
            )

    return rewards


def plot_cumulative_all(monitor_paths: dict[str, str]) -> None:
    """Plot cumulative episode rewards for multiple algorithms in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    plotted = 0
    for algo, path in monitor_paths.items():
        if not os.path.exists(path):
            # Skip missing monitor files
            continue
        ax = axes[plotted]
        rewards = load_monitor_rewards(path)
        cumulative = np.cumsum(rewards)
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative)
        ax.set_title(f"{algo.upper()} – Cumulative Reward", fontsize=11)
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel("Cumulative Reward", fontsize=10)
        ax.grid(alpha=0.4, linestyle="--")
        plotted += 1

    # Hide any unused subplots
    for j in range(plotted, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "all_algorithms_cumulative.png"), dpi=300)
    plt.close()


def plot_training_stability(monitor_path: str, algo_name: str, window: int = 50) -> None:
    """
    Plot a moving-average reward curve to show training stability for a given algorithm.
    """
    episode_rewards = load_monitor_rewards(monitor_path)
    if len(episode_rewards) == 0:
        raise ValueError("No episode rewards found in monitor file.")

    window = min(window, len(episode_rewards))
    kernel = np.ones(window) / window
    smoothed = np.convolve(episode_rewards, kernel, mode="valid")
    x = np.arange(window, window + len(smoothed))

    plt.figure(figsize=(8, 5))
    plt.plot(x, smoothed)
    plt.title(f"{algo_name} – Training Stability (Moving Average, window={window})", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Smoothed Reward", fontsize=12)
    plt.grid(alpha=0.4, linestyle="--")
    plt.tight_layout()
    filename = f"{algo_name.lower().replace(' ', '_')}_training_stability.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def plot_dqn_training_objective(monitor_path: str, window: int = 50) -> None:
    """
    Plot a reward-based training objective curve for DQN using a moving average.
    """
    rewards = load_monitor_rewards(monitor_path)
    if len(rewards) == 0:
        raise ValueError("No episode rewards found in DQN monitor file.")

    window = min(window, len(rewards))
    kernel = np.ones(window) / window
    smoothed = np.convolve(rewards, kernel, mode="valid")
    x = np.arange(window, window + len(smoothed))

    plt.figure(figsize=(8, 5))
    plt.plot(x, smoothed)
    plt.title("DQN – Training Objective (Reward-Based Stability)", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Smoothed Reward", fontsize=12)
    plt.grid(alpha=0.4, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dqn_training_objective.png"), dpi=300)
    plt.close()


def find_convergence_episode(rewards: np.ndarray, window: int = 50, threshold: float = 0.9) -> int:
    """
    Find the episode where the algorithm converges (reaches stable performance).
    
    Convergence is defined as when the moving average reaches threshold (default 90%)
    of the final average performance.
    
    Returns the episode number (1-indexed) where convergence occurs.
    """
    if len(rewards) < window:
        return len(rewards)
    
    # Compute moving average
    kernel = np.ones(window) / window
    smoothed = np.convolve(rewards, kernel, mode="valid")
    x = np.arange(window, window + len(smoothed))
    
    # Final average performance (last 20% of episodes)
    final_window = max(10, len(smoothed) // 5)
    final_avg = np.mean(smoothed[-final_window:])
    
    # Find when smoothed reward reaches threshold of final average
    target = threshold * final_avg
    converged_indices = np.where(smoothed >= target)[0]
    
    if len(converged_indices) > 0:
        # Return the first episode where convergence occurs
        return int(x[converged_indices[0]])
    else:
        # If never reached threshold, return last episode
        return len(rewards)


def plot_convergence_analysis(monitor_paths: dict[str, str], window: int = 50) -> None:
    """
    Plot convergence analysis for all algorithms in a 2x2 subplot grid.
    Shows reward curves with convergence points marked.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    plotted = 0
    convergence_episodes = {}
    
    for algo, path in monitor_paths.items():
        if not os.path.exists(path):
            continue
        
        ax = axes[plotted]
        rewards = load_monitor_rewards(path)
        
        if len(rewards) == 0:
            continue
        
        # Compute moving average
        smooth_window = min(window, len(rewards))
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        x_smooth = np.arange(smooth_window, smooth_window + len(smoothed))
        
        # Find convergence episode
        conv_ep = find_convergence_episode(rewards, window=smooth_window)
        convergence_episodes[algo] = conv_ep
        
        # Plot raw rewards (light, transparent)
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, alpha=0.2, color="gray", linewidth=0.5, label="Raw")
        
        # Plot smoothed curve
        ax.plot(x_smooth, smoothed, linewidth=2, label="Smoothed", color="#2196F3")
        
        # Mark convergence point
        if conv_ep <= len(rewards):
            conv_reward = smoothed[conv_ep - smooth_window] if conv_ep >= smooth_window else rewards[conv_ep - 1]
            ax.axvline(x=conv_ep, color="red", linestyle="--", linewidth=2, label=f"Converged: Ep {conv_ep}")
            ax.plot(conv_ep, conv_reward, "ro", markersize=10, zorder=5)
        
        ax.set_title(f"{algo.upper()} – Convergence Analysis", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Reward", fontsize=11)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9)
        
        plotted += 1
    
    # Hide unused subplots
    for j in range(plotted, len(axes)):
        axes[j].axis("off")
    
    plt.suptitle("Episodes to Converge - Training Stability Analysis", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(OUTPUT_DIR, "convergence_analysis.png"), dpi=300)
    plt.close()
    
    # Print convergence summary
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("="*60)
    for algo, ep in sorted(convergence_episodes.items()):
        print(f"{algo.upper():12} converged at episode {ep:4d}")
    print("="*60 + "\n")


def evaluate_model_generalization(model_or_agent, env_factory, algo_name: str, n_seeds: int = 10, n_episodes_per_seed: int = 5) -> dict:
    """
    Evaluate a model's generalization by testing on multiple random seeds (unseen initial states).
    
    Returns dict with:
    - 'seeds': list of seed values tested
    - 'mean_rewards_per_seed': mean reward for each seed
    - 'std_rewards_per_seed': std reward for each seed
    - 'overall_mean': mean across all seeds
    - 'overall_std': std across all seeds
    """
    if not HAS_RL_IMPORTS:
        return None
    
    seeds = list(range(42, 42 + n_seeds))  # Use seeds 42-51
    mean_rewards_per_seed = []
    std_rewards_per_seed = []
    
    for seed in seeds:
        episode_rewards = []
        
        for _ in range(n_episodes_per_seed):
            if algo_name == "reinforce":
                # REINFORCE evaluation
                env = env_factory()
                obs, _ = env.reset(seed=seed)
                done = False
                episode_reward = 0.0
                
                while not done:
                    obs_tensor = torch.from_numpy(obs).float().to(model_or_agent.device)
                    with torch.no_grad():
                        action_probs = model_or_agent.policy(obs_tensor)
                        dist = torch.distributions.Categorical(action_probs)
                        action = dist.sample()
                    
                    obs, reward, terminated, truncated, _ = env.step(action.item())
                    done = terminated or truncated
                    episode_reward += float(reward)
                
                episode_rewards.append(episode_reward)
            else:
                # SB3 model evaluation
                def make_env_with_seed():
                    e = env_factory()
                    e.reset(seed=seed)
                    return e
                
                vec_env = DummyVecEnv([make_env_with_seed])
                obs = vec_env.reset()
                done = np.array([False])
                episode_reward = 0.0
                
                while not done[0]:
                    action, _ = model_or_agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = vec_env.step(action)
                    done = terminated | truncated
                    episode_reward += float(reward[0])
                
                episode_rewards.append(episode_reward)
                vec_env.close()
        
        mean_rewards_per_seed.append(np.mean(episode_rewards))
        std_rewards_per_seed.append(np.std(episode_rewards))
    
    return {
        'seeds': seeds,
        'mean_rewards_per_seed': mean_rewards_per_seed,
        'std_rewards_per_seed': std_rewards_per_seed,
        'overall_mean': np.mean(mean_rewards_per_seed),
        'overall_std': np.std(mean_rewards_per_seed),
    }


def plot_generalization_analysis(model_paths: dict[str, str]) -> None:
    """
    Plot generalization analysis showing performance variance across unseen initial states.
    Tests each model on multiple random seeds and visualizes the results.
    """
    if not HAS_RL_IMPORTS:
        print("Skipping generalization analysis - RL libraries not available.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    generalization_results = {}
    plotted = 0
    
    for algo, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"Skipping {algo.upper()}: model file not found at {model_path}")
            continue
        
        try:
            print(f"Loading {algo.upper()} model from {model_path}...")
            # Load model
            if algo == "reinforce":
                # Load REINFORCE agent
                env = FoodRedistributionEnv()
                config = ReinforceConfig(episodes=2000, gamma=0.99, lr=1e-3, hidden_dim=64, batch_size=5)
                agent = ReinforceAgent(env.observation_space.shape[0], env.action_space.n, config)
                state_dict = torch.load(model_path, map_location=agent.device)
                agent.policy.load_state_dict(state_dict)
                agent.policy.eval()
                model_or_agent = agent
                env_factory = lambda: FoodRedistributionEnv()
            elif algo == "dqn":
                # Load DQN model (can load without env, but we'll set it later)
                model = DQN.load(model_path)
                model_or_agent = model
                env_factory = lambda: FoodRedistributionEnv()
            elif algo == "ppo":
                # Load PPO model
                model = PPO.load(model_path)
                model_or_agent = model
                env_factory = lambda: FoodRedistributionEnv()
            elif algo == "a2c":
                # Load A2C model
                model = A2C.load(model_path)
                model_or_agent = model
                env_factory = lambda: FoodRedistributionEnv()
            else:
                continue
            
            # Evaluate generalization
            print(f"Evaluating {algo.upper()} generalization on multiple seeds...")
            results = evaluate_model_generalization(model_or_agent, env_factory, algo, n_seeds=10, n_episodes_per_seed=5)
            
            if results is None:
                continue
            
            generalization_results[algo] = results
            ax = axes[plotted]
            
            # Plot mean rewards per seed with error bars
            seeds = results['seeds']
            means = results['mean_rewards_per_seed']
            stds = results['std_rewards_per_seed']
            
            ax.errorbar(seeds, means, yerr=stds, fmt='o-', capsize=5, capthick=2, 
                       linewidth=2, markersize=8, label=f"Mean ± Std per seed")
            ax.axhline(y=results['overall_mean'], color='red', linestyle='--', 
                      linewidth=2, label=f"Overall mean: {results['overall_mean']:.3f}")
            ax.fill_between([seeds[0], seeds[-1]], 
                           results['overall_mean'] - results['overall_std'],
                           results['overall_mean'] + results['overall_std'],
                           alpha=0.2, color='red', label=f"Overall std: {results['overall_std']:.3f}")
            
            ax.set_title(f"{algo.upper()} – Generalization Analysis\n"
                        f"Mean: {results['overall_mean']:.3f} ± {results['overall_std']:.3f}", 
                        fontsize=11, fontweight="bold")
            ax.set_xlabel("Random Seed (Unseen Initial State)", fontsize=10)
            ax.set_ylabel("Mean Episode Reward", fontsize=10)
            ax.grid(alpha=0.3, linestyle="--")
            ax.legend(loc="best", fontsize=9)
            
            plotted += 1
            
        except Exception as e:
            print(f"ERROR: Failed to evaluate {algo.upper()} generalization: {e}")
            print(f"  Model path: {model_path}")
            import traceback
            traceback.print_exc()
            continue
    
    # Hide unused subplots
    for j in range(plotted, len(axes)):
        axes[j].axis("off")
    
    plt.suptitle("Generalization Analysis - Performance on Unseen Initial States", 
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(OUTPUT_DIR, "generalization_analysis.png"), dpi=300)
    plt.close()
    
    # Print generalization summary
    if generalization_results:
        print("\n" + "="*70)
        print("GENERALIZATION ANALYSIS SUMMARY")
        print("="*70)
        print(f"{'Algorithm':<12} {'Mean Reward':<15} {'Std Reward':<15} {'Generalization':<20}")
        print("-"*70)
        for algo, results in sorted(generalization_results.items()):
            gen_quality = "Good" if results['overall_std'] < 0.5 else "Moderate" if results['overall_std'] < 1.0 else "Poor"
            print(f"{algo.upper():<12} {results['overall_mean']:>12.3f}   {results['overall_std']:>12.3f}   {gen_quality:<20}")
        print("="*70 + "\n")


def plot_reinforce_entropy(entropy_path: str) -> None:
    """
    Plot REINFORCE policy entropy over episodes for the best run.
    """
    entropies = load_monitor_rewards(entropy_path)
    if len(entropies) == 0:
        raise ValueError("No entropy values found in REINFORCE entropy file.")

    episodes = np.arange(1, len(entropies) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, entropies)
    plt.title("REINFORCE – Policy Entropy over Episodes", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Entropy", fontsize=12)
    plt.grid(alpha=0.4, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reinforce_entropy.png"), dpi=300)
    plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(CSV_PATHS)

    # Comparison across all algorithms
    plot_mean_reward(df)

    # Monitor files for each algorithm (paths can be adjusted as needed)
    monitor_paths = {
        # Best runs from all_algorithms_summary.json
        "dqn": os.path.join("logs", "dqn_run04_monitor.csv"),
        "ppo": os.path.join("logs", "ppo_run04_monitor.csv"),
        "a2c": os.path.join("logs", "a2c_run01_monitor.csv"),
        # For REINFORCE we use the best run's returns CSV directly
        "reinforce": os.path.join("logs", "reinforce_run07_returns.csv"),
    }

    plot_cumulative_all(monitor_paths)
    plot_training_stability(monitor_paths["ppo"], "PPO", window=50)
    
    # Convergence analysis for all algorithms
    plot_convergence_analysis(monitor_paths, window=50)

    # REINFORCE entropy curve for best run
    reinforce_entropy_path = os.path.join("logs", "reinforce_run07_entropy.csv")
    if os.path.exists(reinforce_entropy_path):
        plot_reinforce_entropy(reinforce_entropy_path)

    # DQN reward-based training objective (use best monitor if available)
    dqn_best_monitor = os.path.join("logs", "dqn_best_monitor.csv")
    if os.path.exists(dqn_best_monitor):
        plot_dqn_training_objective(dqn_best_monitor)
    else:
        # Fallback to best DQN run from summary if alias does not exist
        dqn_fallback = monitor_paths.get("dqn")
        if dqn_fallback and os.path.exists(dqn_fallback):
            plot_dqn_training_objective(dqn_fallback)
    

    print(f"Saved comparison and PPO plots to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

