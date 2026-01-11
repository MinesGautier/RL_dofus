import torch
import os 
import gymnasium as gym

class CommonAgent:
    
    permanent_folder = "SAVED_MODELS_FOLDER"

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        use_action_mask: bool = True,
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
        self.model_paramaters = None
        self.use_action_mask = use_action_mask

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def save(self, filename="latest_model"):
        # Create the folder if it doesn't exist
        if self.model_paramaters == None :
            print("Model is empty for now")
            return
        file_path = "/".join([self.permanent_folder, self.__class__.__name__ + "_" + filename])+".tar"
        os.makedirs(self.permanent_folder, exist_ok=True)
        state = {"state_dict": (self.model_paramaters)}
        torch.save(state, file_path)


    def load(self, filename="latest_model"):
        # 1. Reconstruct the exact same file path used in save()
        file_path = "/".join([self.permanent_folder, self.__class__.__name__ + "_" + filename]) + ".tar"
        
        # 2. Check if the file exists to avoid a crash
        if os.path.exists(file_path):
            # 3. Load the state dictionary from the file
            state = torch.load(file_path)
            
            # 4. Extract and assign the parameters back to the object
            self.model_paramaters = state["state_dict"]
            print(f"Model loaded successfully from {file_path}")
        else:
            print(f"No checkpoint found at {file_path}")

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


if __name__ == "__main__" :
    randomEnv = gym.Env
    my_agent = CommonAgent(randomEnv,
        learning_rate=1,
        initial_epsilon= 1,
        epsilon_decay= 1,
        final_epsilon= 1,)
    my_agent.model_paramaters = [1,1]
    my_agent.save("22h10")
    my_agent.model_paramaters = None
    my_agent.load("22h10")
    print(my_agent.model_paramaters)
    
