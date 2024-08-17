import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb 
import scipy
import torch_dct as freq_transform


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

class HarmonicEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)

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
            SimpleResidualBlock(64, 128, stride=2),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 256, stride=2),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 512, stride=2),
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
        self.harmonic_embedding = HarmonicEmbedding(3, 2)
        self.harmonic_size = 3 * 2 * 2 + 3
        self.feature_branch = nn.Sequential(
            nn.Linear(512+self.harmonic_size, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

        # self.feature_branch = nn.Sequential(
        #     nn.Linear(512, feat_dim),
        # )

    def forward(self, x, desired_pos_ori, return_embedding=False):
        feature = self.encoder(x) 
        harmonic_pos = self.harmonic_embedding(desired_pos_ori)
        feature_output = self.feature_branch(torch.cat((feature, harmonic_pos), dim=1))
        return feature_output







class ResNet18_freq(nn.Module):
    def __init__(self, feat_dim = 8):
        super().__init__()
        self.pos_dim = 3
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1), # try remove maxpool
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 128, stride=2),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 256, stride=2),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 512, stride=2),
            SimpleResidualBlock(512, 512), # 1024
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
        self.harmonic_embedding = HarmonicEmbedding(3, 2)
        self.harmonic_size = 3 * 2 * 2 + 3
        self.feature_branch = nn.Sequential(
            nn.Linear(512+self.harmonic_size, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

        # self.feature_branch = nn.Sequential(
        #     nn.Linear(512, feat_dim),
        # )

    def forward(self, x, desired_pos_ori, return_embedding=False):
        # Convert x to frequency domain
        with torch.no_grad():
            x_freq = freq_transform.dct_2d(x)
        # Concatenate x and x_freq along the channels
        x = torch.cat((x, x_freq), dim=1)
        feature = self.encoder(x) 
        harmonic_pos = self.harmonic_embedding(desired_pos_ori)
        feature_output = self.feature_branch(torch.cat((feature, harmonic_pos), dim=1))
        return feature_output




class ResNet18(nn.Module):
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
            SimpleResidualBlock(64, 128, stride=2),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 256, stride=2),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 512, stride=2),
            SimpleResidualBlock(512, 512), # 1024
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
        self.harmonic_embedding = HarmonicEmbedding(3, 2)
        self.harmonic_size = 3 * 2 * 2 + 3
        self.feature_branch = nn.Sequential(
            nn.Linear(512+self.harmonic_size, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

        # self.feature_branch = nn.Sequential(
        #     nn.Linear(512, feat_dim),
        # )

    def forward(self, x, desired_pos_ori, return_embedding=False):
        feature = self.encoder(x) 
        harmonic_pos = self.harmonic_embedding(desired_pos_ori)
        feature_output = self.feature_branch(torch.cat((feature, harmonic_pos), dim=1))
        return feature_output





class ResNet34_freq(nn.Module):
    def __init__(self, feat_dim = 8):
        super().__init__()
        self.pos_dim = 3
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1), # try remove maxpool
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 128, stride=2),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 256, stride=2),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 512, stride=2),
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
        self.harmonic_embedding = HarmonicEmbedding(3, 2)
        self.harmonic_size = 3 * 2 * 2 + 3
        self.feature_branch = nn.Sequential(
            nn.Linear(512+self.harmonic_size, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

        # self.feature_branch = nn.Sequential(
        #     nn.Linear(512, feat_dim),
        # )

    def forward(self, x, desired_pos_ori, return_embedding=False):
        # Convert x to frequency domain
        with torch.no_grad():
            x_freq = freq_transform.dct_2d(x)
        # Concatenate x and x_freq along the channels
        x = torch.cat((x, x_freq), dim=1)
        feature = self.encoder(x) 
        harmonic_pos = self.harmonic_embedding(desired_pos_ori)
        feature_output = self.feature_branch(torch.cat((feature, harmonic_pos), dim=1))
        return feature_output



class BottleneckResidualBlock(nn.Module):
    expansion = 4  # Expansion ratio used in the bottleneck architecture

    def __init__(self, in_channels, intermediate_channels, stride=1):
        super().__init__()
        out_channels = intermediate_channels * self.expansion

        # First 1x1 conv layer (dimensionality reduction)
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        
        # Second 3x3 conv layer
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        
        # Third 1x1 conv layer (dimensionality expansion)
        self.conv3 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Adjust dimensions if necessary via 1x1 convolution
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Forward pass through the bottleneck
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Add the shortcut before the final ReLU
        out += self.shortcut(identity)
        out = self.relu(out)

        return out
    

import torch
import torch.nn as nn

class BottleneckResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, intermediate_channels, stride=1):
        super().__init__()
        out_channels = intermediate_channels * self.expansion
        
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(identity)
        out = self.relu_out(out)

        return out

class ResNet50_freq(nn.Module):
    def __init__(self, feat_dim=8):
        super().__init__()
        self.pos_dim = 3

        self.encoder = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BottleneckResidualBlock(64, 64),
            BottleneckResidualBlock(256, 64),
            BottleneckResidualBlock(256, 64),
            BottleneckResidualBlock(256, 128, 2),
            BottleneckResidualBlock(512, 128),
            BottleneckResidualBlock(512, 128),
            BottleneckResidualBlock(512, 128),
            BottleneckResidualBlock(512, 256, 2),
            BottleneckResidualBlock(1024, 256),
            BottleneckResidualBlock(1024, 256),
            BottleneckResidualBlock(1024, 256),
            BottleneckResidualBlock(1024, 256),
            BottleneckResidualBlock(1024, 512, 2),
            BottleneckResidualBlock(2048, 512),
            BottleneckResidualBlock(2048, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.pos_branch = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, self.pos_dim)
        )

        self.harmonic_embedding = HarmonicEmbedding(3, 2)
        self.harmonic_size = 3 * 2 * 2 + 3

        self.feature_branch = nn.Sequential(
            nn.Linear(2048 + self.harmonic_size, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

    def forward(self, x, desired_pos_ori, return_embedding=False):
        # Convert x to frequency domain
        with torch.no_grad():
            x_freq = freq_transform.dct_2d(x)
        # Concatenate x and x_freq along the channels
        x = torch.cat((x, x_freq), dim=1)
        feature = self.encoder(x)
        harmonic_pos = self.harmonic_embedding(desired_pos_ori)
        feature_output = self.feature_branch(torch.cat((feature, harmonic_pos), dim=1))
        return feature_output


class ResNet50(nn.Module):
    def __init__(self, feat_dim=8):
        super().__init__()
        self.pos_dim = 3

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BottleneckResidualBlock(64, 64),
            BottleneckResidualBlock(256, 64),
            BottleneckResidualBlock(256, 64),
            BottleneckResidualBlock(256, 128, 2),
            BottleneckResidualBlock(512, 128),
            BottleneckResidualBlock(512, 128),
            BottleneckResidualBlock(512, 128),
            BottleneckResidualBlock(512, 256, 2),
            BottleneckResidualBlock(1024, 256),
            BottleneckResidualBlock(1024, 256),
            BottleneckResidualBlock(1024, 256),
            BottleneckResidualBlock(1024, 256),
            BottleneckResidualBlock(1024, 512, 2),
            BottleneckResidualBlock(2048, 512),
            BottleneckResidualBlock(2048, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.pos_branch = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, self.pos_dim)
        )

        self.harmonic_embedding = HarmonicEmbedding(3, 2)
        self.harmonic_size = 3 * 2 * 2 + 3

        self.feature_branch = nn.Sequential(
            nn.Linear(2048 + self.harmonic_size, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

    def forward(self, x, desired_pos_ori, return_embedding=False):
        feature = self.encoder(x)
        harmonic_pos = self.harmonic_embedding(desired_pos_ori)
        feature_output = self.feature_branch(torch.cat((feature, harmonic_pos), dim=1))
        return feature_output