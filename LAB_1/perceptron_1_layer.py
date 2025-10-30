import numpy as np
import torch
from torch import nn


class Perceptron1Layer(nn.Module):
    def __init__(self, image_size: tuple, num_labels: int):
        super().__init__()
        w, h = image_size
        self.linear = nn.Linear(w * h, num_labels)
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.linear(x)
        output = self.log_softmax(x)
        return output