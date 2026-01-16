import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, FlattenObservation
from envs.mazeEnv import mazeEnv
from stable_baselines3 import DQN


# Training hyperparameters
learning_rate = 0.1  # How fast to learn (higher = faster but less stable)
n_episodes = 100  # Number of episodes to practice
start_epsilon = 1.0  # Start with 100% random actions
epsilon_decay = (start_epsilon / n_episodes) / 3  # Reduce exploration over time
final_epsilon = 0.1  # Always keep some exploration
num_eval_episodes = 10
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
    "exploration_initial_eps": 0.8, 
    "exploration_final_eps": 0.01
}

agent = agent.load("TRAINED_AGENT/DQN_26_01_11-100Mep.zip", custom_objects=custom_params)
agent.set_env(env)

agent.learn(total_timesteps=1_000_000, log_interval=10)
agent.save("TRAINED_AGENT/DQN_26_01_17.zip")


# ## Setting exploration = 0 for pure exploration
# print(f"Starting evaluation for {num_eval_episodes} episodes...")

# for episode_num in range(num_eval_episodes):
#     obs, info = env.reset()
#     episode_reward = 0
#     step_count = 0

#     episode_over = False
#     while not episode_over:
#         # Replace this with your trained agent's policy
#         action = agent.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#         episode_reward += reward
#         step_count += 1

#         episode_over = terminated or truncated

#     print(f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}")

# env.close()

# # Print summary statistics
# print("\nEvaluation Summary:")
# print(f"Episode durations: {list(env.time_queue)}")
# print(f"Episode rewards: {list(env.return_queue)}")
# print(f"Episode lengths: {list(env.length_queue)}")

# # Calculate some useful metrics
# avg_reward = np.sum(env.return_queue)
# avg_length = np.sum(env.length_queue)
# std_reward = np.std(env.return_queue)

# print(f"\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}")
# print(f"Average episode length: {avg_length:.1f} steps")
# print(
#     f"Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}"
# )
