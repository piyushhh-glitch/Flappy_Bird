# 🐦 Flappy Bird — DQN Agent

A Deep Q-Network (DQN) agent that learns to play Flappy Bird on its own using reinforcement learning.

<p align="center">
  <img src="assets/flappy_demo.gif" alt="Flappy Bird DQN Demo" width="600">
</p>

## Project Structure

```text
Flappy_Bird/
├── assets/
│   └── flappy_demo.gif
├── agent.py              # Training/testing loop, epsilon-greedy policy
├── dqn.py                # DQN neural network
├── experience_replay.py  # Replay memory buffer
├── game_flappy_bird.py   # Manual play version
├── parameters.yaml       # Hyperparameters
└── runs/                 # Saved models & logs
```

## How it Works

* Agent observes the game state (bird position, velocity, pipe distances) and picks an action: flap or do nothing
* A neural network (DQN) approximates Q-values for both actions
* Epsilon-greedy strategy: explores randomly early on, gradually shifts to exploiting the learned policy as epsilon decays
* Experience replay stores past transitions and samples random mini-batches for stable training
* A target network (synced periodically) is used to compute stable Q-value targets
* Network is trained to minimize the difference between predicted Q-values and target Q-values (Bellman equation)

## Tech Stack

* **PyTorch** – Neural network & training
* **Gymnasium** – RL environment interface
* **flappy-bird-gymnasium** – Flappy Bird game environment
* **Pygame** – Rendering & manual play
