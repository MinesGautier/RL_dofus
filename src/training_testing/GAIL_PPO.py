import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy


from envs.mazeEnv import mazeEnv
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.data import serialize
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation


SEED = 42

expert_trajectory_path = "EXPERT_TRAJECTORIES"
rollouts = list(serialize.load(expert_trajectory_path))


gym.register(
    id="gymnasium_env/MazeMinogolem-v0",
    entry_point=mazeEnv,
    max_episode_steps=1_000,  # Prevent infinite episodes
)

# Create
env = make_vec_env(
    "gymnasium_env/MazeMinogolem-v0",
    rng=np.random.default_rng(SEED),
    n_envs=1,
    post_wrappers=[
        lambda env, _: RolloutInfoWrapper(env),  # For imitation data collection
        lambda env, _: FlattenObservation(env),  # Fixes the obs_shape = None error
    ],
)

print(f"Current Env Action Space: {env.action_space}")
# Likely prints: Discrete(10)

learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=SEED,
)

reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)


gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=426,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
)

# evaluate the learner before training
env.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    learner,
    env,
    100,
    return_episode_rewards=True,
)

# train the learner and evaluate again
gail_trainer.train(20000)  # Train for 800_000 steps to match expert.
env.seed(SEED)
learner_rewards_after_training, _ = evaluate_policy(
    learner,
    env,
    100,
    return_episode_rewards=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))
