import numpy as np
import torch
from torch import nn


class Perceptron3Layer(nn.Module):
    def __init__(self, image_size: tuple, num_labels: int):
        super().__init__()
        w, h = image_size
        self.layer1 = nn.Linear(w * h, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_labels)
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        x = self.layer1(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        
        x = self.layer3(x)
        output = self.log_softmax(x)
        
        return output