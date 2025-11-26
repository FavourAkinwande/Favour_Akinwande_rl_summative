# Food Redistribution Logistic Agent

A reinforcement learning project that simulates surplus food redistribution in an African urban city. The agent controls a single truck routing food from retailers (supply) to communities (demand) over 24-step days, optimizing for delivery efficiency, freshness, fairness, and waste reduction.

## Project Overview

This project implements and compares four reinforcement learning algorithms:
- **DQN** (Deep Q-Network) - Value-based method
- **PPO** (Proximal Policy Optimization) - Policy-gradient method **Best Performer**
- **A2C** (Advantage Actor-Critic) - Actor-critic method
- **REINFORCE** - Custom policy-gradient baseline

The environment features a discrete action space (9 routes: 3 retailers × 3 communities), complex reward signals, and dynamic supply/demand dynamics.

## Environment Details

### Observation Space

The environment provides a **10-dimensional normalized observation vector** that captures the current state of the food redistribution system:

- **Time feature (1 dim)**: Normalized time step within the 24-step day, ranging from 0.0 (start) to 1.0 (end). This helps the agent understand temporal constraints and plan accordingly.

- **Supply features (3 dims)**: Current food supply at each of the 3 retailers, normalized by `max_supply` (120.0 units). Values range from 0.0 to 1.0, indicating the proportion of maximum capacity remaining at each retailer.

- **Freshness features (3 dims)**: Food freshness level at each retailer, ranging from 0.0 (spoiled) to 1.0 (fresh). Freshness decays over time and affects the freshness bonus in the reward function.

- **Demand features (3 dims)**: Current food demand at each of the 3 communities, normalized by `max_demand` (100.0 units). Values range from 0.0 to 1.0, representing the proportion of maximum demand remaining at each community.

**Observation vector structure**: `[time, supply_R1, supply_R2, supply_R3, freshness_R1, freshness_R2, freshness_R3, demand_C1, demand_C2, demand_C3]`

### Action Space

The environment uses a **discrete action space with 9 possible actions**, where each action represents a delivery route from a retailer to a community:

- **Action encoding**: `action = retailer_idx * num_communities + community_idx`
  - Example: Action 0 = R1→C1, Action 1 = R1→C2, Action 2 = R1→C3
  - Example: Action 3 = R2→C1, Action 4 = R2→C2, Action 5 = R2→C3
  - Example: Action 6 = R3→C1, Action 7 = R3→C2, Action 8 = R3→C3

- **Delivery mechanics**: When an action is taken, the agent attempts to deliver food from the selected retailer to the selected community. The actual amount delivered is constrained by:
  - Truck capacity: 40.0 units per trip
  - Available supply at the retailer
  - Remaining demand at the community
  - Formula: `deliverable = min(truck_capacity, supply, demand)`

### Environment Dynamics

The environment simulates realistic food redistribution dynamics:

- **Supply decay**: After each step, all retailer supplies decrease by 4.0 units (representing spoilage and other losses), encouraging timely deliveries.

- **Freshness decay**: Food freshness decreases by 0.04 per step at all retailers, incentivizing the agent to prioritize fresher food for deliveries.

- **Demand fluctuation**: Community demands fluctuate each step with Gaussian noise (σ=5.0), simulating real-world demand variability. Demands are clipped to remain non-negative and within maximum bounds.

- **Episode structure**: Each episode consists of exactly 24 steps, representing a full day of operations. The episode terminates after 24 steps, regardless of remaining supply or demand.

### Reward Function

The reward function is designed to encourage efficient, fair, and timely food redistribution while penalizing wasteful or inefficient behavior:

**Positive Components:**

- **Delivery Reward**: `deliverable / truck_capacity` (up to +1.0 per step)
  - Rewards successful deliveries proportional to the amount delivered relative to truck capacity
  - Maximum reward occurs when delivering a full truck load (40.0 units)

- **Freshness Bonus**: `delivery_reward * freshness * 0.5` (up to +0.5 per step)
  - Additional reward for delivering fresh food, scaled by the freshness level at the source retailer
  - Encourages prioritizing fresher food to maximize both delivery quantity and quality

**Penalties:**

- **Waste Penalty**: `0.02 * (remaining_supply / max_supply)`
  - Penalizes unused food remaining at retailers, calculated as the proportion of total maximum supply that goes unused
  - Encourages the agent to minimize food waste by delivering as much as possible

- **Fairness Penalty**: `0.2 * std(delivered_ratios)`
  - Penalizes unequal distribution across communities by measuring the standard deviation of delivery ratios
  - Delivery ratio = `delivered_to_community / initial_demand_of_community`
  - Encourages equitable distribution, ensuring all communities receive proportional deliveries relative to their initial needs

- **Transport Penalty**: `0.1 * (distance / max_distance)`
  - Penalizes long-distance deliveries, scaled by the route distance relative to the maximum distance in the network
  - Encourages efficient routing by prioritizing shorter routes when possible
  - Distance matrix: R1→C1=3.0, R1→C2=6.0, R1→C3=9.0, R2→C1=4.0, R2→C2=2.0, R2→C3=7.0, R3→C1=5.0, R3→C2=6.5, R3→C3=3.0

- **Idle Penalty**: `0.1` (fixed penalty per step)
  - Applied when no delivery occurs (e.g., attempting to deliver from an empty retailer or to a community with no demand)
  - Discourages unproductive actions and encourages the agent to make valid deliveries

**Final Reward Formula**: 
```
reward = delivery_reward + freshness_reward - waste_penalty - fairness_penalty - transport_penalty - idle_penalty
```

**Reward Range**: The reward can range from approximately -0.5 (worst case: idle action with high waste) to +1.5 (best case: full truck load of fresh food with minimal penalties). Typical successful episodes achieve cumulative rewards between +1.0 and +4.0.

### Optimization Objectives

The agent must balance multiple competing objectives:

1. **Maximize total deliveries**: Deliver as much food as possible (minimize waste and unmet demand)
2. **Maintain fairness**: Ensure equitable distribution across all communities
3. **Prioritize freshness**: Deliver fresher food when possible to maximize freshness bonuses
4. **Optimize routing**: Minimize transport costs by choosing efficient routes
5. **Time management**: Make productive deliveries within the 24-step constraint

The optimal strategy requires the agent to learn when to prioritize efficiency (high-volume deliveries) versus fairness (balanced distribution), while considering freshness decay and transport costs.

## Training

### Stage 1: Hyperparameter Sweeps (All Algorithms)

Train all four algorithms with 10 hyperparameter combinations each:

```bash
python -m training_scripts.train_all_agents --logs-dir logs --models-dir trained_models
```

**Initial Training Budget:**
- **SB3 algorithms (DQN, PPO, A2C)**: 50,000 timesteps per run
- **REINFORCE**: 2,000 episodes per run

**Output:**
- Model weights saved to `trained_models/`
- Training logs (CSV, JSON) saved to `logs/`
- Best models identified in `logs/all_algorithms_summary.json`

### Stage 2: Extended Training (Best Models)

After identifying the best hyperparameter configurations from Stage 1, the best models are trained for an additional 500,000 timesteps:

```bash
python -m training_scripts.train_best_ppo
```

**Extended Training Budget:**
- **Best PPO model**: 500,000 timesteps (total: 550,000 timesteps)

Saves the final model to `trained_models/best models/overallbest_ppo.zip`

## Results Summary

Based on 50k timestep training sweeps:

| Algorithm | Mean Reward | Std Reward | Best Model |
|-----------|-------------|------------|------------|
| **PPO** | 2.09 | 1.00 | `ppo_run04.zip` |
| **A2C** | 1.67 | 1.08 | `a2c_run01.zip` |
| **REINFORCE** | 0.72 | 1.15 | `reinforce_run07_policy.pt` |
| **DQN** | 0.71 | 1.16 | `dqn_run04.zip` |

**Key Findings:**
- **PPO** achieved the highest mean reward during the initial 50k timestep training sweeps and was selected for extended training (500k timesteps) due to its superior performance
- **A2C** achieved the highest mean reward (2.30) in the final evaluation, demonstrating strong performance with only 50k timesteps of training
- **A2C** showed competitive results with significantly less training, indicating efficient learning
- **DQN** struggled, likely requiring more extensive hyperparameter tuning or longer training
- **REINFORCE** served as an effective baseline, demonstrating the value of variance reduction techniques

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

## Visualization

### Interactive Dashboard

Run the best PPO model with the interactive dashboard:

```bash
python -m training_scripts.play_best_model_dashboard
```

**Features:**
- ModernGL 3D scene with retailers, communities, and animated truck. The scene provides a clear visual representation of the food redistribution network, with retailers displayed as rectangular boxes on the left and communities as cylindrical shapes on the right.
- Real-time episode statistics (supply, demand, rewards, metrics). The dashboard sidebar continuously updates to show current state information, allowing users to monitor the agent's performance and decision-making process in real-time.
- Camera controls (scroll to zoom, drag to rotate). Users can interact with the 3D scene to get different perspectives on the simulation, making it easier to understand the spatial relationships between nodes and delivery routes.
- Episode HUD showing initial supply and total delivered. A semi-transparent overlay in the top-left corner displays key episode metrics, providing quick access to important information without cluttering the main visualization.

### Generate Random Agent Video

Record a video of a random agent:

```bash
python -m training_scripts.random_video --steps 60 --output videos/random_agent.mp4
```

This script generates a video recording of a random policy interacting with the environment, useful for demonstrating the environment dynamics and baseline behavior before training. The video captures the ModernGL-rendered scene, showing how an untrained agent navigates the food redistribution problem.

## Notes

- All algorithms were trained with fixed budgets for fair comparison
- Best models are stored in `trained_models/best models/`
- Training logs include Monitor CSVs (SB3) and returns/entropy CSVs (REINFORCE)
- Evaluation results are saved in `model_evaluation/evaluation_results.csv`
- Comprehensive summaries available in `logs/all_algorithms_summary.json`

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
