from typing import Any


import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from envs.mazeEnv import mazeEnv
from stable_baselines3 import PPO
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt


# Training hyperparameters
training_timesteps = 1e5
recording_period = training_timesteps / 100

gym.register(
    id="gymnasium_env/MazeMinogolem-v0",
    entry_point=mazeEnv,
    max_episode_steps=1_000,  # Prevent infinite episodes
)
# Create environment and agent
# Configuration
env_name = "gymnasium_env/MazeMinogolem-v0"  # Replace with your environment

env = gym.make(
    env_name, render_mode="rgb_array"
)  # rgb_array needed for video recording

# Add video recording for every episode
env = RecordVideo(
    env,
    video_folder="MazeMinogolem-agent",  # Folder to save videos
    name_prefix="eval",  # Prefix for video filenames
    episode_trigger=lambda x: x % recording_period == 0,  # Record every episode
)

# Add episode statistics tracking
# env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

agent = PPO("MultiInputPolicy", env, verbose=1)

custom_params = {
    "exploration_fraction": 0.5,  # Decay over 50% of the NEW 100k steps
    "exploration_initial_eps": 0.5,
    "exploration_final_eps": 0.01,
}

# Create log directory
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor[Any, Any](env, log_dir)

agent = agent.load("TRAINED_AGENT/DQN_26_01_17.zip", custom_objects=custom_params)
agent.set_env(env)

agent.learn(total_timesteps=training_timesteps, log_interval=5)
agent.save("TRAINED_AGENT/PPO_26_01_27.zip")

plot_results([log_dir], 20_000, results_plotter.X_EPISODES, "Minogolem")
plt.show()

env.close()
