import numpy as np

from unityagents import UnityEnvironment


class BananaEnv():
    """
    Unity Banana environment simulation
    """

    def __init__(self, path='Banana_Windows_x86_64/Banana.exe', seed=None, 
            verbose=False
        ):
        """
        Class constructor / BananaEnv initializer

        Args:
            path: String path to Unity Banana simulation
            seed: Integer representing seed for random number generation
            verbose: Boolean specifying verbosity
        
        Returns:
            BananaEnv class object
        
        """

        # Initialize environment variables
        self.env = None
        self.brain_name = None
        self.brain = None
        self.agents = None
        self.action_size = None
        self.state_size = None
        self.rng = None
        self.rng_seed = None
        self.verbose = verbose

        # Initialize environment status variables    
        self.env_info = None
        self.state = None
        self.reward = None
        self.done = None
        
        # Create environment
        self.env = UnityEnvironment(file_name=path, seed=seed)

        # Get default Unity environment "brain"
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # Set seed
        self.seed(seed)

        # Reset environment
        self.reset()

        return

    def reset(self, train=False):
        """
        Reset environment

        Args:
            train: Boolean specifying training mode
        
        Returns:
            None
        
        """

        # Reset environment
        self.env_info = self.env.reset(train_mode=train)[self.brain_name]

        # Set environment variables
        self.agents = self.env_info.agents
        self.action_size = self.brain.vector_action_space_size
        self.state = self.env_info.vector_observations[0]
        self.state_size = len(self.state)

        # Create list of allowable actions
        self.actions = list(range(self.action_size))
        self.actions.append(None)

        # Debug
        if self.verbose:
            print('Number of agents: {}'.format(len(self.agents)))
            print('Number of actions: {}'.format(self.action_size))
            print('Valid actions: {}'.format(self.actions))
            print('States look like: {}'.format(self.state))
            print('States have length: {}'.format(self.state_size))

        return

    def step(self, action=None):
        """
        Perform specified action in environment. If no action is specified then
        sample & perform random action

        Args:
            action: Integer representing action to be performed

        Returns:
            Tuple containing environment state, reward, brain info, & 
            completion status
            
            state: Vector representing environment observation 
            reward: Float representing reward for performing specified action
            info: BrainInfo object representing latest environment information
            done: Boolean representing if simulation is complete
        
        """
                
        # Error check
        if action not in self.actions:
            raise ValueError('Invalid action specified. Valid options are: '
                '{}'.format(self.actions)
            )

        # Sample random action if no action specified
        if action is None:
            action = self.rng.choice(self.actions)

        # Send action to environment
        self.env_info = self.env.step(action)[self.brain_name]  

        # Get environment status
        self.state = self.env_info.vector_observations[0]
        self.reward = self.env_info.rewards[0]
        self.done = self.env_info.local_done[0]

        return self.state, self.reward, self.env_info, self.done

    def close(self):
        """
        Close environment

        Args:
            None

        Returns:
            None
        
        """
        self.env.close()
        return

    def seed(self, seed=None):
        """
        Set seed for random number generation, sampling, & repeatibility
        
        Args:
            seed: Integer representing seed for random number generation

        Returns:
            None
        
        """

        # Error check
        if not isinstance(seed, int) and seed is not None:
            raise ValueError('Specified seed must be integer or None')

        # Set seed & random number generator
        self.rng_seed = seed
        self.rng = np.random.RandomState(seed)

        return
