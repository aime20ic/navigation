import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent():
    """Deep Q-Learning agent that interacts with & learns from environment"""

    def __init__(self, state_size, action_size, seed, verbose=False, **kwargs):
        """Initialize an Agent object
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            verbose (bool): verbosity

        Returns:
            None
        
        """

        # Set agent class variables
        self.t_step = 0
        self.rng = None
        self.rng_seed = None
        self.verbose = verbose

        # Set environment variables
        self.state_size = state_size
        self.action_size = action_size
        
        # Set agent training hyperparameters
        self.hidden = kwargs.get('hidden', [256,128,64])        # hidden layer architecture
        self.buffer_size = kwargs.get('buffer_size', int(1e5))  # replay buffer size
        self.batch_size = kwargs.get('batch_size', 64)          # minibatch size
        self.gamma = kwargs.get('gamma', 0.99)                  # discount factor
        self.tau = kwargs.get('tau', 1e-3)                      # for soft update of target parameters
        self.lr = kwargs.get('lr', 5e-4)                        # learning rate 
        self.update_every = kwargs.get('update_every', 4)       # how often to update the network     
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Set seed
        self.seed(seed)

        # Set agent q-networks & optimizer
        self.qnetwork_local = QNetwork(
            self.state_size, self.action_size, self.rng_seed, self.hidden
        ).to(self.device)
        self.qnetwork_target = QNetwork(
            self.state_size, self.action_size, self.rng_seed, self.hidden
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(
            action_size=self.action_size, 
            buffer_size=self.buffer_size, 
            batch_size=self.batch_size, 
            seed=self.rng_seed,
            device=self.device
        )

        return
    
    def step(self, state, action, reward, next_state, done):
        """
        Add memory to experience replay buffer & learn if required

        Args:
            state (vector): Environment observation before action
            action (int): Action performed
            reward (float): Reward for performing specified action
            next_state (vector): Environment observation after action
            done (bool): Is simulation complete
        
        Returns:
            None
        
        """

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every self.update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every

        # Get random subset & learn if enough samples available in memory
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
        
        return

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy
        
        Args:
            state (array_like): Current state
            eps (float): Epsilon, for epsilon-greedy action selection

        Returns:
            None
        
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if self.rng.uniform() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return self.rng.choice(np.arange(self.action_size))
        
        return

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples

        Args:
            experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done) tuples 
            gamma (float): Discount factor
        
        Returns:
            None

        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)  

        return                   

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (PyTorch model): Weights will be copied from
            target_model (PyTorch model): Weights will be copied to
            tau (float): Interpolation parameter 
        
        Returns:
            None
        """
        
        # Get parameters
        parameters = zip(target_model.parameters(), local_model.parameters())

        # Update target model parameters
        for target_param, local_param in parameters:
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data
            )

        return

    def seed(self, seed=None):
        """
        Set seed for random number generation, sampling, & repeatibility
        
        Args:
            seed (int): Seed for random number generation

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

