from typing import Any


import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo, FlattenObservation
from envs.mazeEnv import mazeEnv
from stable_baselines3 import DQN
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt


# Training hyperparameters
learning_rate = 0.1  # How fast to learn (higher = faster but less stable)
n_episodes = 100  # Number of episodes to practice
start_epsilon = 1.0  # Start with 100% random actions
epsilon_decay = (start_epsilon / n_episodes) / 3  # Reduce exploration over time
final_epsilon = 0.1  # Always keep some exploration
num_eval_episodes = 0
recording_period = 10_000

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

agent = DQN("MultiInputPolicy", env, verbose=1,exploration_final_eps=0.01)

custom_params = {
    "exploration_fraction": 0.5,  # Decay over 50% of the NEW 100k steps
    "exploration_initial_eps": 0.2, 
    "exploration_final_eps": 0.01
}

# Create log directory
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor[Any, Any](env, log_dir)


agent = agent.load("TRAINED_AGENT/DQN_26_01_17.zip", custom_objects=custom_params)
agent.set_env(env)

agent.learn(total_timesteps=100_000, log_interval=5)
agent.save("TRAINED_AGENT/DQN_26_01_27.zip")

plot_results([log_dir], 20_000, results_plotter.X_EPISODES, "Minogolem")

plt.show()

env.close()

