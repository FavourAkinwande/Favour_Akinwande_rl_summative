# Food Redistribution Logistic Agent

A reinforcement learning project that simulates surplus food redistribution in an African urban city. The agent controls a single truck routing food from retailers (supply) to communities (demand) over 24-step days, optimizing for delivery efficiency, freshness, fairness, and waste reduction.

## Project Overview

This project implements and compares four reinforcement learning algorithms:
- **DQN** (Deep Q-Network) - Value-based method
- **PPO** (Proximal Policy Optimization) - Policy-gradient method **Best Performer**
- **A2C** (Advantage Actor-Critic) - Actor-critic method
- **REINFORCE** - Custom policy-gradient baseline

The environment features a discrete action space (9 routes: 3 retailers × 3 communities), complex reward signals, and dynamic supply/demand dynamics.

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for visualization
pip install moderngl Pillow
```

## Training

### Hyperparameter Sweeps (All Algorithms)

Train all four algorithms with 10 hyperparameter combinations each:

```bash
python -m training_scripts.train_all_agents --logs-dir logs --models-dir trained_models
```

**Training Budget:**
- **SB3 algorithms (DQN, PPO, A2C)**: 50,000 timesteps per run
- **REINFORCE**: 2,000 episodes per run

**Output:**
- Model weights saved to `trained_models/`
- Training logs (CSV, JSON) saved to `logs/`
- Best models identified in `logs/all_algorithms_summary.json`

### Extended PPO Training

Train the best PPO configuration for 500,000 timesteps:

```bash
python -m training_scripts.train_best_ppo
```

Saves the model to `trained_models/best models/overallbest_ppo.zip`

## Evaluation

### Comprehensive Model Evaluation

Evaluate all trained models over 30 episodes:

```bash
python -m model_evaluation.evaluation
```

**Output:**
- `model_evaluation/evaluation_results.csv` - Detailed metrics
- `logs/*_summary.json` - Per-run summaries
- Console output with comparison table and analysis

**Metrics Computed:**
- Mean reward, standard deviation, max, min
- Success rate (reward > 0)
- Stability score (variance)
- Episode length statistics

### Generate Performance Visualizations

Create all performance plots:

```bash
python -m model_evaluation.performance_plots
```

**Generated Plots** (saved to `figures/`):
- `mean_reward_errorbars.png` - Algorithm comparison with error bars
- `all_algorithms_cumulative.png` - Cumulative rewards (2×2 subplots)
- `ppo_training_stability.png` - PPO smoothed reward curve
- `dqn_training_objective.png` - DQN reward-based stability
- `reinforce_entropy.png` - REINFORCE policy entropy over episodes
- `convergence_analysis.png` - Episodes to converge for all algorithms

## Visualization

### Interactive Dashboard

Run the best PPO model with the interactive dashboard:

```bash
python -m training_scripts.play_best_model_dashboard
```

**Features:**
- ModernGL 3D scene with retailers, communities, and animated truck
- Real-time episode statistics (supply, demand, rewards, metrics)
- Camera controls (scroll to zoom, drag to rotate)
- Episode HUD showing initial supply and total delivered

### Generate Random Agent Video

Record a video of a random agent:

```bash
python -m training_scripts.random_video --steps 60 --output videos/random_agent.mp4
```

## Results Summary

Based on comprehensive evaluation:

| Algorithm | Mean Reward | Std Reward | Success Rate | Best Model |
|-----------|-------------|------------|--------------|------------|
| **PPO** | 2.24 | 0.73 | 100% | `overallbest_ppo.zip` |
| **A2C** | 2.30 | 0.74 | 100% | `a2c_run01.zip` |
| **REINFORCE** | 0.59 | 1.11 | 70% | `reinforce_run07_policy.pt` |
| **DQN** | -2.63 | 0.38 | 0% | `dqn_run04.zip` |

**Key Findings:**
- **PPO** achieved the highest mean reward (2.09) during 50k timestep training sweeps
- **PPO** was selected for extended training (500k timesteps) due to superior performance
- **A2C** showed competitive results but with 10× less training
- **DQN** struggled, likely requiring more extensive hyperparameter tuning or longer training
- **REINFORCE** served as an effective baseline, demonstrating the value of variance reduction techniques

## Environment Details

### Observation Space
- **10 normalized features**: `[time, supplies(3), freshness(3), demands(3)]`
- Time: normalized step (0-1) within the 24-step day
- Supplies/Demands: normalized by max values
- Freshness: decay factor (0-1)

### Action Space
- **9 discrete actions**: Each represents a route from retailer (R1-R3) to community (C1-C3)
- Action encoding: `action = retailer_idx * num_communities + community_idx`

### Reward Function

**Positive Components:**
- **Delivery Reward**: `deliverable / truck_capacity` (up to +1.0)
- **Freshness Bonus**: `delivery_reward * freshness * 0.5` (up to +0.5)

**Penalties:**
- **Waste Penalty**: `0.02 * (remaining_supply / max_supply)`
- **Fairness Penalty**: `0.2 * std(delivered_ratios)` - encourages equitable distribution
- **Transport Penalty**: `0.1 * (distance / max_distance)` - discourages long routes
- **Idle Penalty**: `0.1` if no delivery occurs

**Final Reward**: `delivery_reward + freshness_reward - waste_penalty - fairness_penalty - transport_penalty - idle_penalty`

## Technical Details

### Algorithms Implemented

**DQN (Stable-Baselines3)**
- Network: 2-layer MLP [128, 128]
- Features: Experience replay, target networks, epsilon-greedy exploration
- Hyperparameters: Learning rate, buffer size, batch size, gamma, exploration schedule

**PPO (Stable-Baselines3)**
- Network: 2-layer MLP [128, 128]
- Features: Clipped objective, multiple optimization epochs
- Hyperparameters: Learning rate, n_steps, batch_size, gamma, n_epochs

**A2C (Stable-Baselines3)**
- Network: 2-layer MLP [128, 128]
- Features: Advantage estimation, on-policy updates
- Hyperparameters: Learning rate, n_steps, gamma, entropy coefficient

**REINFORCE (Custom PyTorch)**
- Network: 2-layer MLP with configurable hidden dimension
- Features: Vanilla policy gradient, batch updates
- Hyperparameters: Learning rate, gamma, hidden_dim, batch_size

### Visualization Technologies

- **Pygame**: UI shell, dashboard, and controls
- **ModernGL (OpenGL)**: 3D scene rendering with retailers, communities, truck, and routes
- **Pillow**: Text overlay and labels on rendered frames

## Notes

- All algorithms were trained with fixed budgets for fair comparison
- Best models are stored in `trained_models/best models/`
- Training logs include Monitor CSVs (SB3) and returns/entropy CSVs (REINFORCE)
- Evaluation results are saved in `model_evaluation/evaluation_results.csv`
- Comprehensive summaries available in `logs/all_algorithms_summary.json`

## Project Structure

```
Food_Redistribution_Logistic-Agent/
├── environment/
│   └── food_env.py              # Custom Gymnasium environment
├── training_scripts/
│   ├── train_all_agents.py      # Hyperparameter sweeps for all algorithms
│   ├── train_best_ppo.py        # Extended training for best PPO model
│   ├── reinforce.py              # Custom REINFORCE implementation
│   ├── play_best_model_dashboard.py  # Interactive visualization
│   ├── play_model.py             # Simple model playback
│   └── random_video.py           # Generate random agent video
├── visualization_scripts/
│   ├── food_dashboard.py         # Pygame dashboard with ModernGL integration
│   ├── opengl_scene.py           # ModernGL 3D scene renderer
│   └── pygame_viewer.py         # Basic pygame viewer
├── model_evaluation/
│   ├── evaluation.py             # Comprehensive model evaluation
│   ├── performance_plots.py     # Visualization generation
│   └── evaluation_results.csv    # Evaluation metrics
├── logs/                         # Training logs and summaries
│   ├── *_monitor.csv            # Episode rewards (SB3)
│   ├── *_summary.json           # Per-run summaries
│   ├── all_algorithms_summary.json  # Combined results
│   └── reinforce_*_returns.csv  # REINFORCE episode returns
│   └── reinforce_*_entropy.csv   # REINFORCE policy entropy
├── trained_models/              # Saved model weights
│   └── best models/             # Best performing models
├── figures/                      # Generated visualization plots
└── videos/                       # Recorded simulation videos
```

## References

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **ModernGL**: https://moderngl.readthedocs.io/

## Author

University assignment project for reinforcement learning course.

---

**Best Model**: `trained_models/best models/overallbest_ppo.zip` (PPO, 500k timesteps)
