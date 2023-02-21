from torch import nn
from project_model import Model
from torch.functional import F


class QNetwork(Model):
    def __init__(self, num_channels, num_out, hidden_size):
        super().__init__(num_channels, num_out)
        self.fc1 = nn.Linear(4 * 4 * 32, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_out)
    
    def fc_forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
