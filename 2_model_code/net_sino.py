import torch, torch.nn as nn
import numpy as np

class Estimate_Noise_FC(nn.Module):
    def __init__(self, in_ch=1, h_ch=64, kernel_size=5):
        super().__init__()

        self.fc1 = nn.Linear(kernel_size**2, h_ch)
        self.fc2 = nn.Linear(h_ch, h_ch)
        self.fc3 = nn.Linear(h_ch, h_ch)
        self.fc4 = nn.Linear(h_ch, 1)
        self.relu = nn.ReLU()

        self.kernel_size = kernel_size
        self.flatten = nn.Flatten()
        
        self.res = nn.Linear(kernel_size**2, 1)
        self.res.weight.data.fill_(0.0)
        self.res.weight.data[0][kernel_size**2//2].fill_(1.0)
        self.res.bias.data.fill_(0.0)
        for param in self.res.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.flatten(x)
        res = self.res(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x) + res)
        return x
