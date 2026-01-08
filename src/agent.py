from collections import defaultdict
import gymnasium as gym
import numpy as np
import os
import torch


class mazeAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.q_value_invalid_move = -1e9

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def _mask_invalid_actions(self, observation, valid_actions):
        for i in range(len(self.q_values[observation])):
            if i not in valid_actions:
                self.q_values[observation][i] = self.q_value_invalid_move

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 1 to n
        """
        # With probability epsilon: explore (random action)
        obs = tuple(obs)
        valid_actions = self.env.unwrapped.get_possible_actions()
        if np.random.random() < self.epsilon:
            i = int(np.random.random() * len(valid_actions))
            return valid_actions[i]
        # With probability (1-epsilon): exploit (best known action)
        else:
            self._mask_invalid_actions(obs, valid_actions)
            x = int(np.argmax(self.q_values[obs]))
            return x

    def save(self, filename="latest_model"):
        # Create the folder if it doesn't exist
        permanent_folder = "SAVED_MODELS_FOLDER"
        file_path = permanent_folder + "/" + filename + ".tar"
        os.makedirs(permanent_folder, exist_ok=True)
        state = {"state_dict": dict(self.q_values)}
        torch.save(state, file_path)

    def load(self, filepath):
        saved_model_q_values = torch.load(filepath)
        self.q_values = saved_model_q_values

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        obs = tuple(obs)
        next_obs = tuple(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
