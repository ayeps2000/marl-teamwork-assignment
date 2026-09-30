"""Check that your installation works. Run this first: python check_setup.py"""

import sys
import time
from importlib.metadata import version

if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    print(f"Warning: use Python 3.10, 3.11 or 3.12; you have {sys.version.split()[0]}.")

for package in ["pettingzoo", "mpe2", "supersuit", "stable-baselines3", "torch"]:
    print(f"{package:18s} {version(package)}")

import gymnasium as gym
import supersuit  # noqa: F401  (only checks that it imports)
from stable_baselines3 import PPO

from marl.envs import make_env

# 1. Play one episode with random actions.
env = make_env()
observations, _ = env.reset(seed=0)
steps = 0
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, _, _, _ = env.step(actions)
    steps += 1
env.close()
print(f"\nPlayed one random episode: {steps} steps.")

# 2. Train PPO for a few seconds on a single-agent task to check the training libraries.
model = PPO("MlpPolicy", gym.make("CartPole-v1"), n_steps=512, batch_size=64, verbose=0)
start = time.time()
model.learn(total_timesteps=3_000)
print(f"Trained PPO for {model.num_timesteps} timesteps in {time.time() - start:.1f} s.")
print("\nSetup OK.")
