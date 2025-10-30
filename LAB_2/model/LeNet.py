import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2)
        )
        
        self.pool1 = nn.AvgPool2d(
            kernel_size=(2, 2),
            stride=2
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5),
            stride=1,
            padding=0
        )   
        
        self.pool2 = nn.AvgPool2d(
            kernel_size=(2, 2),
            stride=2
        )
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x = x.unsqueeze(1)  
        
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = self.pool2(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.sigmoid(x)
        
        x = self.fc2(x)
        x = F.sigmoid(x)
        
        output = self.output(x)
        
        return output
        