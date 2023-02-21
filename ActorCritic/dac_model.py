import torch
import torch.nn as nn
import torch.nn.functional as F

from project_model import Model

class DACModel(Model):
    def __init__(self, num_channels, num_out, gamma=0.99, lr=0.0001, device='cpu'):
        super(DACModel, self).__init__(num_channels, num_out)
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device

        # Define the actor and critic networks
        self.actor = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_out),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

    def fc_forward(self, X):
        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X
    def actor_loss(actions, advantages):
        log_probs = torch.log(actions)
        return -(log_probs * advantages).mean()

    def critic_loss(rewards, values, next_values, done_mask):
        target = rewards + (1 - done_mask) * self.gamma * next_values
        td_error = target - values
        return td_error.pow(2).mean()
