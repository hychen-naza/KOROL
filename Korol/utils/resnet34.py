import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb 
import scipy


import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb 

class SimpleResidualBlock(nn.Module):
    def __init__(self, input_channel_size, out_channel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel_size, out_channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel_size, out_channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel_size)
        if stride == 1:
            if (input_channel_size == out_channel_size):
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Conv2d(input_channel_size, out_channel_size, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(input_channel_size, out_channel_size, kernel_size=1, stride=stride),
                                        nn.Conv2d(out_channel_size, out_channel_size, kernel_size=1, stride=stride))
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out = self.relu2(out + shortcut)        
        return out

class ResNet34(nn.Module):
    def __init__(self, feat_dim = 8):
        super().__init__()
        self.pos_dim = 3
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1), # try remove maxpool
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 128, 2),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 256, 2),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 512, 2),
            SimpleResidualBlock(512, 512), # 1024
            SimpleResidualBlock(512, 512), 
            nn.AdaptiveAvgPool2d((1, 1)), # For each channel, collapses (averages) the entire feature map (height & width) to 1x1
            nn.Flatten(), # the above ends up with batch_size x 512 x 1 x 1, flatten to batch_size x 512
            # nn.Linear(512, 128),
            # nn.ReLU(),
            # nn.Linear(128, feat_dim)
        )

        self.pos_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.pos_dim)
        )

        self.feature_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

    def forward(self, x, return_embedding=False):
        feature = self.encoder(x) 
        pos_output = self.pos_branch(feature)
        feature_output = self.feature_branch(feature)
        return pos_output, feature_output
