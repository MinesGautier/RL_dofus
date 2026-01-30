from typing import Any


import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from envs.mazeEnv import mazeEnv
from stable_baselines3 import DQN
import os
from stable_baselines3.common.monitor import Monitor
from imitation.data import serialize


# Training hyperparameters
training_timesteps = 5e7
recording_period = training_timesteps / 200

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

# Create log directory
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor[Any, Any](env, log_dir)

# Add video recording for every episode
env = RecordVideo(
    env,
    video_folder="agent-training-recording",  # Folder to save videos
    name_prefix="eval",  # Prefix for video filenames
    episode_trigger=lambda x: x % recording_period == 0,  # Record every episode
    disable_logger=True,
)
# Add episode statistics tracking
# env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

agent = DQN("MultiInputPolicy", env, verbose=1)

custom_params = {
    "exploration_fraction": 0.5,  # Decay over 50% of the NEW 100k steps
    "exploration_initial_eps": 0.5,
    "exploration_final_eps": 0.2,
}

agent.load("TRAINED_AGENT/DQN_26_01_29.zip", custom_objects=custom_params)

agent.learn(total_timesteps=training_timesteps, log_interval=10)
agent.save("TRAINED_AGENT/DQN_26_01_29.zip")


env.close()
