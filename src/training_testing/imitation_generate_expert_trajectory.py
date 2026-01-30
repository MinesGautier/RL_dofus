from typing import Any
from envs.mazeEnv import mazeEnv
from imitation.data.rollout import TrajectoryAccumulator
from imitation.data import serialize
from gymnasium.wrappers import FlattenObservation
from utils import read_single_keypress

expert_trajectory_path = "EXPERT_TRAJECTORIES"

# --- Setup ---
env = mazeEnv(render_mode="human")  # Test with human mode first
# env = RolloutInfoWrapper(env) # Required by 'imitation' to track metadata
env = FlattenObservation(env)

# 2. Generate the Trajectories
# 'min_episodes' tells it how many successful runs you want to record


db_trajectories = []
try:
    db_trajectories = list(serialize.load(expert_trajectory_path))
except:
    print(f"0 trajectory in DB for now")


this_run_trajectories = []
# accumulator for incomplete trajectories
trajectories_accum = TrajectoryAccumulator()
obs, info = env.reset()
trajectory_key = 0
trajectories_accum.add_step({"obs": obs}, key=trajectory_key)

shortcuts_commands = {
    "d": 0,
    "z": 1,
    "q": 2,
    "s": 3,
    " ": 5,
    "D": 6,
    "Z": 7,
    "Q": 8,
    "S": 9,
}

while True:
    env.render()
    # Collect rollout tuples.
    val = read_single_keypress("Enter action : ")
    if val in shortcuts_commands.keys():
        print(
            f"transforming {val} into {shortcuts_commands[val]} using shortcuts_commands"
        )
        val = shortcuts_commands[val]
    try:
        action = int(val)
    except:
        val2 = read_single_keypress("Are you sure you want to exit the game (y/n)? ")
        if val2 == "y":
            val3 = read_single_keypress("Do you want to save this trajectory (y/n)? ")
            if val3 == "y":
                this_run_trajectories.append(
                    trajectories_accum.finish_trajectory(trajectory_key, terminal=False)
                )
            break
        elif val2 == "n":
            pass
    trajectories_accum.add_step({"obs": obs, "acts": action}, trajectory_key)
    new_obs, rew, done, _, _ = env.step(action)
    trajectories_accum.add_step({"rews": float(rew), "infos": 0}, trajectory_key)
    obs = new_obs

    if done:
        this_run_trajectories.append(
            trajectories_accum.finish_trajectory(trajectory_key, terminal=True)
        )
        db_trajectories.extend(this_run_trajectories)
        this_run_trajectories = []
        print(f"Adding 1 trajectory in DB")
        serialize.save(expert_trajectory_path, db_trajectories)
        trajectory_key += 1
        obs, _ = env.reset()
        trajectories_accum.add_step({"obs": obs}, key=trajectory_key)


# 3. Save the Data
print("End of the process: Saving all trajectories")
db_trajectories.extend(this_run_trajectories)

print(f"Adding {len(this_run_trajectories)} trajectories in DB")
serialize.save(expert_trajectory_path, db_trajectories)

# 4. Verif that I'm not loosing previous data
print(f"db_trajectories now has {len(db_trajectories)} trajectories in it")
for i in range(len(db_trajectories)):
    traj = db_trajectories[i]
    print(
        f"---Traj #{i + 1}--- \n nb of actions = {len(traj.acts)} \n total reward = {sum(traj.rews)} ? "
    )
