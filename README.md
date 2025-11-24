# Food Redistribution Logistic Agent

Custom reinforcement learning environment that simulates surplus food redistribution in an African urban city. Agents must route a single truck from retailers (supply) to communities (demand) over a 24-step day while maximizing total deliveries, freshness, fairness, transport efficiency, and waste reduction.

## Project Structure

- `environment/food_env.py` — Gymnasium-compatible environment with rendering support
- `train_agents.py` — trains DQN, PPO, A2C (Stable-Baselines3) and a custom REINFORCE agent
- `reinforce.py` — lightweight PyTorch REINFORCE implementation
- `visualization/pygame_viewer.py` — realtime pygame visualization driven by a random agent
- `scripts/generate_random_video.py` — saves an MP4 of a random policy interacting with the env

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Training

```bash
python train_agents.py --timesteps 20000 --episodes 500 --models-dir models
```

Outputs are stored inside `models/` (Stable-Baselines `.zip` files plus `reinforce_policy.pt`).

## Visualization & Video

- Interactive pygame viewer:
  ```bash
  python visualization/pygame_viewer.py --fps 3
  ```
- Random agent video:
  ```bash
  python scripts/generate_random_video.py --output videos/random_agent.mp4
  ```

The viewer uses the environment's `render(mode="human")` path, while the video script records the `rgb_array` rendering pipeline.

## Notes

- Observation space: 10 normalized features `[time, supplies(3), freshness(3), demand(3)]`
- Action space: 9 discrete actions representing retailer→community routes
- Rewards blend delivery success, freshness bonus, fairness penalty, transport cost, and waste reduction