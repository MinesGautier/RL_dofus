import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, FlattenObservation
from mazeEnv import MazeEnv
from agent import mazeAgent

# Training hyperparameters
learning_rate = 0.1  # How fast to learn (higher = faster but less stable)
n_episodes = 100  # Number of episodes to practice
start_epsilon = 1.0  # Start with 100% random actions
epsilon_decay = (start_epsilon / n_episodes) / 3  # Reduce exploration over time
final_epsilon = 0.1  # Always keep some exploration
training_period = 1_000

gym.register(
    id="gymnasium_env/MazeMinogolem-v0",
    entry_point=MazeEnv,
    max_episode_steps=1_000,  # Prevent infinite episodes
)
# Create environment and agent
env = gym.make("gymnasium_env/MazeMinogolem-v0", render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
wrapped_env = FlattenObservation(env)

# Record videos periodically (every xx episodes)
wrapped_env = RecordVideo(
    wrapped_env,
    video_folder="MazeMinogolem-training",
    name_prefix="training",
    episode_trigger=lambda x: x % training_period
    == 0,  # Only record every 250th episode
)

agent = mazeAgent(
    env=wrapped_env,
    learning_rate=1,
    initial_epsilon=0,
    epsilon_decay=0,
    final_epsilon=0,
)
agent.load("/Users/gautier/Documents/RL_minogolem/SAVED_MODELS_FOLDER/latest_model.tar")


# Configuration
num_eval_episodes = 4
env_name = "gymnasium_env/MazeMinogolem-v0"  # Replace with your environment

# Create environment with recording capabilities
env = gym.make(
    env_name, render_mode="rgb_array"
)  # rgb_array needed for video recording

# Add video recording for every episode
env = RecordVideo(
    env,
    video_folder="MazeMinogolem-agent",  # Folder to save videos
    name_prefix="eval",  # Prefix for video filenames
    episode_trigger=lambda x: x % 10 == 0,  # Record every episode
)

# Add episode statistics tracking
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

print(f"Starting evaluation for {num_eval_episodes} episodes...")
print("Videos will be saved to: cartpole-agent/")

for episode_num in range(num_eval_episodes):
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0

    episode_over = False
    while not episode_over:
        # Replace this with your trained agent's policy
        action = agent.get_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1

        episode_over = terminated or truncated

    print(f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}")

env.close()

# Print summary statistics
print("\nEvaluation Summary:")
print(f"Episode durations: {list(env.time_queue)}")
print(f"Episode rewards: {list(env.return_queue)}")
print(f"Episode lengths: {list(env.length_queue)}")

# Calculate some useful metrics
avg_reward = np.sum(env.return_queue)
avg_length = np.sum(env.length_queue)
std_reward = np.std(env.return_queue)

print(f"\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}")
print(f"Average episode length: {avg_length:.1f} steps")
print(
    f"Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}"
)
