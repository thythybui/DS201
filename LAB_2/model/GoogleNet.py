import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
    
class InceptionModule(nn.Module):
    def __init__(self,channels,c1,c2,c3,c4,c5,c6: int):
        super().__init__()
        
        #left
        self.conv_left = nn.Conv2d(in_channels=channels, out_channels=c1, kernel_size=1, padding=0)
        
        #middle
        self.conv_1_1 = nn.Conv2d(in_channels=channels, out_channels=c2, kernel_size=1)
        self.conv_1_2 = nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, padding=1)
        
        self.conv_2_1 = nn.Conv2d(in_channels=channels, out_channels=c4, kernel_size=1)
        self.conv_2_2 = nn.Conv2d(in_channels=c4, out_channels=c5, kernel_size=5, padding=2)
        
        #right
        self.maxpool_right = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
        self.conv_right = nn.Conv2d(in_channels=channels, out_channels=c6, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = F.relu(self.conv_left(x))
        
        middle_1 = F.relu(self.conv_1_1(x))
        middle_1 = F.relu(self.conv_1_2(middle_1))
        
        middle_2 = F.relu(self.conv_2_1(x))
        middle_2 = F.relu(self.conv_2_2(middle_2))
        
        right = self.maxpool_right(x)
        right = F.relu(self.conv_right(right))
        
        output = torch.cat([left, middle_1, middle_2, right], dim=1)
        
        return output
        
        
class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.inception_3a = InceptionModule(channels=192, c1=64, c2=96, c3=128, c4=16, c5=32, c6=32)
        self.inception_3b = InceptionModule(channels=256, c1=128, c2=128, c3=192, c4=32, c5=96, c6=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.inception_4a = InceptionModule(channels=480, c1=192, c2=96, c3=208, c4=16, c5=48, c6=64)
        self.inception_4b = InceptionModule(channels=512, c1=160, c2=112, c3=224, c4=24, c5=64, c6=64)   
        self.inception_4c = InceptionModule(channels=512, c1=128, c2=128, c3=256, c4=24, c5=64, c6=64)
        self.inception_4d = InceptionModule(channels=512, c1=112, c2=144, c3=288, c4=32, c5=64, c6=64)
        self.inception_4e = InceptionModule(channels=528, c1=256, c2=160, c3=320, c4=32, c5=128, c6=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.inception_5a = InceptionModule(channels=832, c1=256, c2=160, c3=320, c4=32, c5=128, c6=128)
        self.inception_5b = InceptionModule(channels=832, c1=384, c2=192, c3=384, c4=48, c5=128, c6=128)   
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=1000)
        self.output = nn.Softmax(dim=1)
        
    
    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool2(x)
        
        x = self.inception_3a.forward(x)
        x = self.inception_3b.forward(x)
        x = self.maxpool3(x)
        
        x = self.inception_4a.forward(x)
        x = self.inception_4b.forward(x)
        x = self.inception_4c.forward(x)
        x = self.inception_4d.forward(x)
        x = self.inception_4e.forward(x)
        x = self.maxpool4(x)
        
        x = self.inception_5a.forward(x)
        x = self.inception_5b.forward(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        output = self.output(x)
        
        return output