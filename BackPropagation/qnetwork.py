import torch
import torch.nn as nn
import torch.nn.functional as F

from project_model import Model

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, channels=4, fc1_units=2592, fc2_units=256):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.channels = channels

        self.model = Model(num_channels=self.channels, num_out=fc1_units)
        self.fc1 = nn.Linear(fc1_units, fc2_units)
        self.fc2 = nn.Linear(fc2_units, action_size)

    def conv_forward(self, X):
        X = self.model.conv_forward(X)
        X = X.view(X.size(0), -1)
        return X

    def fc_forward(self, X: torch.Tensor) -> torch.Tensor:
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X

    def forward(self, X):
        print("X was ", X.shape)
        X = X.permute(3, 1, 2, 0) # Transpose input tensor to have channels dimension as the second dimension
        # X = X.permute(0, 3, 1, 2)
        print("X is ", X.shape)
        X = self.conv_forward(X)
        print("X after conv_forward is ", X.shape)
        X = self.fc_forward(X)
        print("X after fc_forward is ", X.shape)
        return X

