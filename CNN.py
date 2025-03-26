import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

class ConvNeuralNet(nn.module):
    
    def __init__(self, in_channels, num_classes):
        super(ConvNeuralNet(), self).__init__()

        self.conv1 = nn.conv2d(in_channels = in_channels, out_channels = 8, kernel_size=3)
        self.conv2 = nn.conv2d(in_channels = 8, out_channels = 16, kernel_size=3)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = nn.conv2d(in_channels = 16, out_channels = 32, kernel_size=3)
        self.conv4 = nn.conv2d(in_channels = 32, out_channels = 64, kernel_size=3)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2,stride=2) 

    
    #def forward_prop(self, x):
