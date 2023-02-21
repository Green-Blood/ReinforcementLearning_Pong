from importlib.machinery import OPTIMIZED_BYTECODE_SUFFIXES
from math import tau
from pickletools import optimize
import random
import gym
import numpy as np
import torch
from BackPropagation.qnetwork import QNetwork
from BackPropagation.replay_buffer import ReplayBuffer
from torch.functional import F
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3, lr=5e-4, update_every=4):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr (float): learning rate
            update_every (int): how often to update the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.t_step = 0
        
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """       
        state = np.array(state)
        state = torch.from_numpy(state).float() # Transpose the input tensor to have channels dimension as the second dimension
        if(state.dim == 3):
            state.unsqueeze(0)
         
        # state = torch.from_numpy(state).transpose((2, 0, 1)).unsqueeze(0).float()
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    
    
    def step(self, state, action, reward, next_state, done):
        
        # next_state = np.array(next_state)
        # next_state = np.rollaxis(next_state, 3, 1)
        # next_state = np.rollaxis(next_state, 2, 1)
        # next_state = next_state.reshape(1, 4, 84, 84) 
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done)
        
        print("Shape of the next exp in step before step", next_state.shape)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                print("Shape of the next exp in step before sampling", next_state.shape)
                
                experiences = self.memory.sample()
                
                print("Shape of the next step in step agent ", next_state.shape)
                print("Shape of the next exp in step ", experiences[3].shape)

                self.learn(experiences)
    
    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            state (torch.Tensor): current state
            action (int): chosen action
            reward (float): reward received
            next_state (torch.Tensor): next state
            done (bool): whether episode finished
        """
        states, actions, rewards, next_states, dones = experiences
        print("Shape of the next step in learn ", next_states.shape)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0]

        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

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
    
    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
