import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__ (self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1
                            )
        
        self.bn_1 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_2 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3, 
                                stride=1, 
                                padding=1
                            )
        
        self.bn_2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, 
                          out_channels, 
                          kernel_size=1, 
                          stride=stride
                        ),
                nn.BatchNorm2d(out_channels)
            )
            
def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        
        super(ResNet18, self).__init__()
        
        self.in_channels = 64
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(InceptionModule, 64, 2, stride=1)
        self.layer2 = self._make_layer(InceptionModule, 128, 2, stride=2)
        self.layer3 = self._make_layer(InceptionModule, 256, 2, stride=2)
        self.layer4 = self._make_layer(InceptionModule, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        