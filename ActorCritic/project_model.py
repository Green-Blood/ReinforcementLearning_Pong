import torch
from torch import nn
from torch.functional import F

class Model(nn.Module):
    def __init__(self, num_channels, num_out) -> None:
        super(Model, self).__init__()   
        # DeepMind ATARI(MNIH ET AL) CONVNET Paper
        self.conv0 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=8, stride=4) # 16 8x8 stride=4
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2) #32 4x4 stride=2
        
        self.fc1 = nn.Linear(in_features=2592, out_features=256)  # 32 * 9 * 9 = 2592
        self.fc2 = nn.Linear(in_features=256, out_features=num_out) 
        
    def conv_forward(self, X):
        X = F.relu(self.conv0(X))
        X = F.relu(self.conv1(X))
        return X
    
    def fc_forward(self, X: torch.Tensor) -> torch.Tensor:           
        pass

    def forward(self, X):                                
        X = self.conv_forward(X)
        X = self.fc_forward(X)
        return X

