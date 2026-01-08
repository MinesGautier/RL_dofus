import matplotlib

matplotlib.use("qtagg")
from typing import Optional
import numpy as np
import gymnasium as gym
import random as rand
from joueur import joueur
import cv2


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    maze_observation_size = 9
    penalty_invalid_move = -100
    penalty_backward_move = -1
    reward_forward_move = 5
    reward_useful_push = 10
    reward_valid_push = 3
    penalty_useless_boost = -10
    reward_distance_traveled = 10
    reward_smarter_than_minogolem = 30
    penalty_dying = -5
    penalty_stuck = -10
    penalty_dumber_than_minogolem = -10
    reward_win = 100

    initial_maze = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        ]
    )

    initial_start = (18, 0)
    my_exit = (0, 18)

    PATTERNS = {
        1: ["Sel", "Or", "Sang", "Or"],
        2: ["Or", "Sang", "Sel", "Sang"],
        3: ["Sang", "Sel", "Or", "Sel"],
        4: ["Or", "Sel", "Sang", "Sel"],
        5: ["Sel", "Sang", "Or", "Sang"],
        6: ["Sang", "Or", "Sel", "Or"],
    }

    POSITION_INITIALE_MINOGOLEM = {
        "Séculaire": "haut",
        "Or": "gauche",
        "Sang": "bas",
        "Sel": "droite",
    }

    def __init__(self, render_mode):
        super(MazeEnv, self).__init__()
        # The size of the square grid
        self.maze = self.initial_maze.copy()
        self.size = self.maze.shape
        self.render_mode = render_mode

        self.__ax1 = None

        # Code jeu
        self.joueur = joueur()
        self.tour = 1
        self.current_minogolem_positions = self.POSITION_INITIALE_MINOGOLEM.copy()
        self.current_patterns = []
        self.translated_patterns = []

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._initial_agent_location = self.initial_start
        self._agent_location = self.initial_start
        self._target_location = self.my_exit

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    0, self.size[0] - 1, shape=(2,), dtype=int
                ),  # [x, y] coordinates
                "target": gym.spaces.Box(
                    0, self.size[1] - 1, shape=(2,), dtype=int
                ),  # [x, y] coordinates
                "maze_observation": gym.spaces.Box(
                    low=0,
                    high=2,
                    shape=(self.maze_observation_size, self.maze_observation_size),
                    dtype=int,
                ),
                "PV": gym.spaces.Box(0, self.joueur.PV_initiaux, shape=(1,), dtype=int),
                "PM": gym.spaces.Box(0, 10, shape=(1,), dtype=int),
                "Relance_boost_PM": gym.spaces.Box(0, 10, shape=(1,), dtype=int),
                "Relance_pousse": gym.spaces.Box(0, 10, shape=(1,), dtype=int),
                "Minogolems_next_plays": gym.spaces.Box(-1, 3, shape=(8,), dtype=int),
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(10)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: {
                "direction": np.array([0, 1]),
                "nom": "Déplacement droite",
            },  # Move right (column + 1)
            1: {
                "direction": np.array([-1, 0]),
                "nom": "Déplacement haut",
            },  # Move up (row - 1)
            2: {
                "direction": np.array([0, -1]),
                "nom": "Déplacement gauche",
            },  # Move left (column - 1)
            3: {"direction": np.array([1, 0]), "nom": "Déplacement bas"},
            4: {"nom": "Boost PM"},
            5: {"nom": "Passe tour"},
            6: {
                "direction": np.array([0, 1]),
                "nom": "Pousse droite",
            },  # Move right (column + 1)
            7: {
                "direction": np.array([-1, 0]),
                "nom": "Pousse haut",
            },  # Move up (row - 1)
            8: {
                "direction": np.array([0, -1]),
                "nom": "Pousse gauche",
            },  # Move left (column - 1)
            9: {"direction": np.array([1, 0]), "nom": "Pousse bas"},
        }

        self._minogolem_position_to_direction = {
            "haut": {
                "direction": np.array([1, 0]),
                "obs_code": 3,
            },  # placer un bloc en dessous
            "bas": {
                "direction": np.array([-1, 0]),
                "obs_code": 1,
            },  # placer un bloc au dessus
            "gauche": {
                "direction": np.array([0, 1]),
                "obs_code": 0,
            },  # placer un bloc à droite
            "droite": {
                "direction": np.array([0, -1]),
                "obs_code": 2,
            },  # placer un bloc à gauche
        }

    def get_possible_actions(self):
        res = [5]
        ##Déplacements##
        for i in range(4):
            if (
                self.what_block_is_here(
                    self._agent_location + self._action_to_direction[i]["direction"]
                )
                == 1
            ):
                res.append(i)
        if self.joueur.relance_boost_PM == 0:
            res.append(4)
        if self.joueur.relance_pousse == 0:
            for i in range(6, 10):
                if (
                    self.what_block_is_here(
                        self._agent_location + self._action_to_direction[i]["direction"]
                    )
                    == 2
                ):
                    if (
                        self.what_block_is_here(
                            self._agent_location
                            + 2 * self._action_to_direction[i]["direction"]
                        )
                        == 1
                    ):
                        res.append(i)
        return res

    def read_minogolems(self):
        res = np.array([-1] * 8, dtype=int)
        patterns_copy = self.current_patterns.copy()
        current_minogolem_positions_copy = self.current_minogolem_positions.copy()
        i = 0
        while patterns_copy:
            next_golem = patterns_copy.pop()
            block_direction_code = self._minogolem_position_to_direction[
                current_minogolem_positions_copy[next_golem]
            ]["obs_code"]
            res[i] = block_direction_code
            ##Rotate current_minogolem_positions
            next_golem_position = current_minogolem_positions_copy[next_golem]
            current_minogolem_positions_copy[next_golem] = (
                current_minogolem_positions_copy["Séculaire"]
            )
            current_minogolem_positions_copy["Séculaire"] = next_golem_position
            i += 1
        return res

    def what_block_is_here(self, position):
        (nrows, ncols) = self.size
        if self.render_mode == "human":
            print(f"Evaluating what block is here : {position}")
        if (
            position[0] < 0
            or position[0] >= nrows
            or position[1] < 0
            or position[1] >= ncols
        ):  # If we are looking outside the maze
            return 0  # Return 0 as if it was a wall
        else:
            return self.maze[position[0], position[1]]

    def generate_pattern(self):
        number = rand.randint(1, 6)
        # if self.render_mode == "human":
        for i in range(3, -1, -1):
            self.current_patterns.append(self.PATTERNS[number][i])

    def minogolem_play(self):
        if self.current_patterns:
            next_golem = self.current_patterns.pop()
            if self.render_mode == "human":
                print(f"The golem {next_golem} now plays")
            block_direction = self._minogolem_position_to_direction[
                self.current_minogolem_positions[next_golem]
            ]["direction"]
            block_position = self._agent_location + block_direction
            if (
                self.what_block_is_here(block_position) == 1
            ):  # Golem puts a wall inside an empty cell
                self.maze[block_position[0], block_position[1]] = 2
                if self.render_mode == "human":
                    print(
                        f"The golem {next_golem} puts a block in {(int(block_position[0]), int(block_position[1]))}"
                    )
            else:  # Golem puts a wall in a wall
                if self.render_mode == "human":
                    print(f"The golem {next_golem} puts a block in the wall")
            ##Rotate current_minogolem_positions
            next_golem_position = self.current_minogolem_positions[next_golem]
            self.current_minogolem_positions[next_golem] = (
                self.current_minogolem_positions["Séculaire"]
            )
            self.current_minogolem_positions["Séculaire"] = next_golem_position
        else:
            if self.render_mode == "human":
                print("The minogolem doesn't play now")

    def nombre_vides_adjacents(self, position):
        res = 0
        for i in range(4):
            direction = self._action_to_direction[i]["direction"]
            if self.what_block_is_here(position + direction) == 1:
                res += 1
        return res

    def est_couloir(self, position):
        return self.nombre_vides_adjacents(position) == 2

    def est_carrefour(self, position):
        return self.nombre_vides_adjacents(position) > 2

    def next_golem_move_reward_or_penalty(self):
        reward = 0
        if self.current_patterns:
            next_golem = self.current_patterns[-1]
            block_direction = self._minogolem_position_to_direction[
                self.current_minogolem_positions[next_golem]
            ]["direction"]
            next_block_position = self._agent_location + block_direction
            if (block_direction == np.array([-1, 0])).all() or (
                block_direction == np.array([0, 1])
            ).all():
                if self.what_block_is_here(next_block_position) != 1:
                    reward += self.reward_smarter_than_minogolem
                    if self.render_mode == "human":
                        print("You outplayed the minogolem")
                elif self.maze[
                    next_block_position[0], next_block_position[1]
                ] == 1 and self.est_couloir(self._agent_location):
                    reward += self.penalty_dumber_than_minogolem
                    if self.render_mode == "human":
                        print("You have been outplayed by the minogolem")
        return reward

    def maze_observation(self):
        maze_observation = np.zeros(
            (self.maze_observation_size, self.maze_observation_size), dtype=int
        )
        (nrows, ncols) = self.size
        observation_radius = int(self.maze_observation_size / 2)
        for i in range(self.maze_observation_size):
            for j in range(self.maze_observation_size):
                maze_i = self._agent_location[0] + i - observation_radius
                maze_j = self._agent_location[1] + j - observation_radius
                if maze_i >= 0 and maze_i < nrows and maze_j >= 0 and maze_j < ncols:
                    maze_observation[i, j] = self.maze[maze_i, maze_j]
        return maze_observation

    def obs_minogolem_positions(self):
        position_to_numbers = {"droite": 0, "haut": 1, "gauche": 2, "bas": 3}
        result = []
        result.append(
            position_to_numbers[self.current_minogolem_positions["Séculaire"]]
        )
        result.append(position_to_numbers[self.current_minogolem_positions["Sang"]])
        result.append(position_to_numbers[self.current_minogolem_positions["Or"]])
        result.append(position_to_numbers[self.current_minogolem_positions["Sel"]])
        return np.array(result)

    def obs_minogolem_patterns(self):
        minogolem_to_numbers = {"Séculaire": 0, "Sang": 1, "Or": 2, "Sel": 3}
        result = [-1] * 8
        if self.current_patterns:
            for i in range(len(self.current_patterns)):
                try:
                    result[i] = minogolem_to_numbers[self.current_patterns[i]]
                except:
                    print(
                        f"self.tour = {self.tour} ; current state : i = {i}, result = {result}, current_patterns = {self.current_patterns}"
                    )
                    return error
        return np.array(result)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {
            "agent": np.array(self._agent_location),
            "target": np.array(self._target_location),
            "maze_observation": self.maze_observation(),
            "PV": np.array(self.joueur.PV),
            "PM": np.array(self.joueur.PM),
            "Relance_boost_PM": np.array(self.joueur.relance_boost_PM),
            "Relance_pousse": np.array(self.joueur.relance_pousse),
            "Minogolems_next_plays": self.read_minogolems(),
        }

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                np.subtract(self._agent_location, self._target_location), ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self.render()

        self._agent_location = self._initial_agent_location
        self.maze = self.initial_maze.copy()
        self.joueur.reset()
        self.current_patterns = []
        self.current_minogolem_positions = self.POSITION_INITIALE_MINOGOLEM.copy()
        self.tour = 1

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def manual_step(self):
        while True:
            self.render()
            print("What the agent will see")
            print(self.maze_observation())
            ## Show information for player
            print(f"#### Début du tour {self.tour} ####")
            print("\n### informations minogolems ###")
            print(f"Current patterns = {self.current_patterns[::-1]}")
            print("Current minogolem positions:")
            inv_minogolem_positions = {
                v: k for k, v in self.current_minogolem_positions.items()
            }
            largeur_ligne = 20
            print(
                f"{'-' * int((largeur_ligne - len(inv_minogolem_positions['haut'])) / 2)}{inv_minogolem_positions['haut']}{'-' * int((largeur_ligne - len(inv_minogolem_positions['haut'])) / 2)}"
            )
            print(
                f"{inv_minogolem_positions['gauche']}{(largeur_ligne - len(inv_minogolem_positions['gauche']) - len(inv_minogolem_positions['droite'])) * '-'}{inv_minogolem_positions['droite']}"
            )
            print(
                f"{'-' * int((largeur_ligne - len(inv_minogolem_positions['bas'])) / 2)}{inv_minogolem_positions['bas']}{'-' * int((largeur_ligne - len(inv_minogolem_positions['bas'])) / 2)}"
            )
            n = len(self.current_patterns)
            for i in range(n):
                print(f"{self.current_patterns[n - 1 - i]} : dans {i + 1} tour(s)")
            print("\n### informations joueurs ###")
            print(f"Nombre de PM disponible(s) : {self.joueur.PM}")
            print(f"Nombre de PV : {self.joueur.PV}")
            print(f"Relance sort boost PM : {self.joueur.relance_boost_PM}")
            print(f"Relance sort pousse : {self.joueur.relance_pousse}")
            for i in range(len(self._action_to_direction)):
                print(f"taper {i} pour : {self._action_to_direction[i]['nom']}")

            ## Play input
            action = int(input())
            print(self.step(action))

    def is_stuck(self):
        if self.nombre_vides_adjacents(self._agent_location) != 0:
            return False
        else:
            for i in range(4):
                if (
                    self.what_block_is_here(
                        self._agent_location + self._action_to_direction[i]["direction"]
                    )
                    == 2
                    and self.what_block_is_here(
                        self._agent_location
                        + 2 * self._action_to_direction[i]["direction"]
                    )
                    == 1
                ):
                    return False
            return True

    def usefull_boost_pm(self):
        res = 0
        if self.joueur.relance_boost_PM == 10 and self.joueur.PM > 0:
            if self.render_mode == "human":
                print(
                    f"Penalty: This was a useless boost since you didn't use all the PM : reward+= {self.penalty_useless_boost}"
                )
            res += self.penalty_useless_boost
        return res

    def pousse(self, direction):
        res = 0
        if self.joueur.relance_pousse > 0:
            if self.render_mode == "human":
                print(f"can't push because relance is {self.joueur.relance_pousse}")
            res += self.penalty_invalid_move
        else:
            aiming_at = self._agent_location + direction
            final_block_position = self._agent_location + 2 * direction
            if (self.what_block_is_here(aiming_at) == 2) and self.what_block_is_here(
                final_block_position
            ) == 1:
                if self.render_mode == "human":
                    print(
                        f"Pushing the block from {aiming_at} to {final_block_position}"
                    )
                self.maze[aiming_at[0], aiming_at[1]] = 1
                self.maze[final_block_position[0], final_block_position[1]] = 2
                self.joueur.relance_pousse = 10
                if self.est_carrefour(aiming_at):
                    res += self.reward_useful_push
                    if self.render_mode == "human":
                        print("this push is considered a successful push ")
                else:
                    res += self.reward_valid_push
                    if self.render_mode == "human":
                        print(
                            "this push is valid but is not considered a successful push "
                        )
            else:
                if self.render_mode == "human":
                    print(f"can't push {aiming_at} to {final_block_position}")
                res += self.penalty_invalid_move
        return res

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        reward = 0
        if action < 4:
            if self.joueur.peut_avancer():
                direction = self._action_to_direction[action]["direction"]
                next_position = self._agent_location + direction
                if self.render_mode == "human":
                    print(f"Agent is trying to go in {next_position}")
                (nrows, ncols) = self.size
                # Update agent position, ensuring it stays within grid bounds
                # np.clip prevents the agent from walking off the edge
                if (
                    next_position[0] < 0
                    or next_position[0] >= nrows
                    or next_position[1] < 0
                    or next_position[1] >= ncols
                ):
                    reward += self.penalty_invalid_move
                else:
                    if self.maze[next_position[0], next_position[1]] == 1:
                        self._agent_location = next_position
                        self.joueur.avancer()
                        if action == 0 or action == 1:
                            reward += self.reward_forward_move
                            if self.render_mode == "human":
                                print(
                                    f"Bravo tu as avancé : + {self.reward_forward_move} pour Gryffondor"
                                )
                    else:
                        reward += self.penalty_invalid_move
            else:
                if self.render_mode == "human":
                    print("Le joueur n'a plus de PM")
                reward += self.penalty_invalid_move
        elif action == 4:
            if self.joueur.relance_boost_PM == 0:
                self.joueur.boost_PM()
            else:
                reward += self.penalty_invalid_move

        elif action == 5:
            reward += self.next_golem_move_reward_or_penalty()
            reward = +self.usefull_boost_pm()
            self.joueur.passe_tour()
            self.tour += 1
            self.minogolem_play()
            if self.tour == 2:
                self.generate_pattern()
                self.generate_pattern()
            elif self.tour > 2 and self.tour % 4 == 2:
                self.generate_pattern()

        elif action >= 6:
            reward += self.pousse(self._action_to_direction[action]["direction"])

        terminated = False

        # Check if agent is stuck
        if self.is_stuck():
            reward += self.penalty_stuck
            terminated = True
        # Check if agent is dead
        if self.joueur.estMort():
            reward += self.penalty_dying
            terminated = True

        # Check if agent reached the target
        if np.array_equal(self._agent_location, self._target_location):
            reward += self.reward_win
            terminated = True

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            print(f"observation : {observation}")
            print(f"reward : {reward}")
            print(f"terminated : {terminated}")
            print(f"truncated : {truncated}")
            print(f"info : {info}")

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment for human viewing."""
        if self.render_mode == "rgb_array":
            # 1. Define visual parameters (to match your 600x600 Plotly output)
            target_size = 600
            nrows, ncols = self.size
            cell_size = target_size // max(nrows, ncols)

            # 2. Create the base maze image (Black background)
            # Start with a grayscale image based on maze (0=wall/black, 1=path/white)
            # We multiply by 255 to get the white path color
            maze_img = (self.maze * 255).astype(np.uint8)

            # Scale up the pixels to 'cell_size'
            img = np.repeat(np.repeat(maze_img, cell_size, axis=0), cell_size, axis=1)

            # Convert to RGB (OpenCV uses BGR by default, so we use (255, 255, 255))
            rgb_array = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # 3. Draw the Grid (Black lines)
            # Vertical lines
            for i in range(ncols + 1):
                x = i * cell_size
                cv2.line(
                    rgb_array, (x, 0), (x, rgb_array.shape[0]), (0, 0, 0), thickness=2
                )

            # Horizontal lines
            for i in range(nrows + 1):
                y = i * cell_size
                cv2.line(
                    rgb_array, (0, y), (rgb_array.shape[1], y), (0, 0, 0), thickness=2
                )

            special_cells = np.argwhere(self.maze == 2)
            for row, col in special_cells:
                pt1 = (col * cell_size, row * cell_size)
                pt2 = ((col + 1) * cell_size, (row + 1) * cell_size)
                # OpenCV uses BGR: (Blue, Green, Red) -> (0, 255, 0) is Green
                cv2.rectangle(rgb_array, pt1, pt2, (0, 255, 0), thickness=-1)

            # 4. Draw the Agent (Blue Circle)
            # Coordinates in OpenCV are (x, y) which is (col, row)
            row, col = self._agent_location
            center_x = int((col + 0.5) * cell_size)
            center_y = int((row + 0.5) * cell_size)
            radius = int(cell_size * 0.3)

            # Draw blue circle (RGB: 0, 0, 255)
            cv2.circle(
                rgb_array,
                (center_x, center_y),
                radius,
                (0, 0, 255),
                -1,
                lineType=cv2.LINE_AA,
            )

            # 5. Ensure final size is exactly 600x600 (optional)
            rgb_array = cv2.resize(
                rgb_array, (target_size, target_size), interpolation=cv2.INTER_NEAREST
            )

            return rgb_array

        if self.render_mode == "human":
            print("Entering render function in human mode ")
            ## Show information for player
            print(f"#### Début du tour {self.tour} ####")
            print("\n### informations minogolems ###")
            print(f"Current patterns = {self.current_patterns[::-1]}")
            print("Current minogolem positions:")
            inv_minogolem_positions = {
                v: k for k, v in self.current_minogolem_positions.items()
            }
            largeur_ligne = 20
            print(
                f"{'-' * int((largeur_ligne - len(inv_minogolem_positions['haut'])) / 2)}{inv_minogolem_positions['haut']}{'-' * int((largeur_ligne - len(inv_minogolem_positions['haut'])) / 2)}"
            )
            print(
                f"{inv_minogolem_positions['gauche']}{(largeur_ligne - len(inv_minogolem_positions['gauche']) - len(inv_minogolem_positions['droite'])) * '-'}{inv_minogolem_positions['droite']}"
            )
            print(
                f"{'-' * int((largeur_ligne - len(inv_minogolem_positions['bas'])) / 2)}{inv_minogolem_positions['bas']}{'-' * int((largeur_ligne - len(inv_minogolem_positions['bas'])) / 2)}"
            )
            n = len(self.current_patterns)
            for i in range(n):
                print(f"{self.current_patterns[n - 1 - i]} : dans {i + 1} tour(s)")
            print("\n### informations joueurs ###")
            print(f"Nombre de PM disponible(s) : {self.joueur.PM}")
            print(f"Nombre de PV : {self.joueur.PV}")
            print(f"Relance sort boost PM : {self.joueur.relance_boost_PM}")
            print(f"Relance sort pousse : {self.joueur.relance_pousse}")
            for i in range(len(self._action_to_direction)):
                print(f"taper {i} pour : {self._action_to_direction[i]['nom']}")

            # Print a simple ASCII representation
            for x in range(self.size[0]):  # Top to bottom
                row = ""
                for y in range(self.size[1]):
                    if np.array_equal([x, y], self._agent_location):
                        row += "A "  # Agent
                    elif np.array_equal([x, y], self._target_location):
                        row += "T "  # Target
                    elif self.maze[x, y] == 1:
                        row += ". "  # Empty
                    elif self.maze[x, y] == 0:
                        row += "/ "  # Wall
                    elif self.maze[x, y] == 2:
                        row += "+ "  # Moveable wall
                print(row)


# TEST##

# my_env = MazeEnv(render_mode="rgb_array")
# my_env.render()
# my_env.manual_step()


#####REGISTER

# # Register the environment so we can create it with gym.make()
# gym.register(
#     id="gymnasium_env/MazeMinogolem-v0",
#     entry_point=MazeEnv,
#     max_episode_steps=300,  # Prevent infinite episodes
# )

# # Create the environment like any built-in environment
# env = gym.make("gymnasium_env/MazeMinogolem-v0")

# # Create the environment like any built-in environment
# gym.pprint_registry()


# import gymnasium as gym


# from gymnasium.utils.env_checker import check_env

# # This will catch many common issues
# try:
#     check_env(MazeEnv)
#     print("Environment passes all checks!")
# except Exception as e:
#     print(f"Environment has issues: {e}")
