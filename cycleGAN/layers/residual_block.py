import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, kernel_size=3, padding=1, padding_mode='reflect'):
        super(ResidualBlock, self).__init__()
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        y = x
        y = self.activation(self.instancenorm(self.conv1(y)))
        y = self.instancenorm(self.conv2(y))
        return y + x