from collections import deque
import numpy as np
import torch


class ExperienceReplay:
    def __init__(self, buffer_size):
        self.capacity = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*random.sample(self.buffer, batch_size))
        return torch.stack(state_batch), torch.tensor(action_batch), torch.tensor(reward_batch), torch.stack(next_state_batch), torch.tensor(done_batch)
    
    def __len__(self):
        return len(self.buffer)
    
    def add_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
