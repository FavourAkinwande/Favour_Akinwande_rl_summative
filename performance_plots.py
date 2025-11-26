import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def plot_success_vs_stability(df: pd.DataFrame) -> None:
    algorithms = df["algorithm"]
    success_rates = df["success_rate"]
    stabilities = df["stability_score"]

    x = np.arange(len(algorithms))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, success_rates, width, label="Success Rate", color="#00BCD4")
    plt.bar(x + width / 2, stabilities, width, label="Stability Score", color="#FF7043")
    plt.xticks(x, algorithms)
    plt.title("Success Rate vs Stability Score", fontsize=14)
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "success_vs_stability.png"), dpi=300)
    plt.close()


def plot_radar(df: pd.DataFrame) -> None:
    algorithms = df["algorithm"]
    metrics = ["mean_reward", "success_rate", "stability_score", "efficiency_score"]

    max_episode_length = df["mean_episode_length"].max()
    df["efficiency_score"] = 1 - (df["mean_episode_length"] / max_episode_length)

    data = df[metrics].to_numpy()
    max_values = data.max(axis=0)
    normalized = data / max_values

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for idx, algo in enumerate(algorithms):
        values = normalized[idx]
        values = np.concatenate([values, [values[0]]])
        ax.plot(angles, values, linewidth=2, label=algo)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        ["Mean Reward", "Success Rate", "Stability Score", "Efficiency"], fontsize=11
    )
    ax.set_yticklabels([])
    ax.set_title("Algorithm Comparison (Radar Chart)", fontsize=15, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "radar_comparison.png"), dpi=300)
    plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(CSV_PATHS)

    plot_mean_reward(df)
    plot_success_vs_stability(df)
    plot_radar(df)

    print(f"Saved plots to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

