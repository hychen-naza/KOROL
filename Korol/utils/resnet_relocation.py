import torch.nn as nn
import torch 
import pdb 
import torch_dct as freq_transform

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

class ClassificationNetwork18(nn.Module):
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
            SimpleResidualBlock(64, 128, 2),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 256, 2),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 512, 2),
            SimpleResidualBlock(512, 512), # 1024
            nn.AdaptiveAvgPool2d((1, 1)), # For each channel, collapses (averages) the entire feature map (height & width) to 1x1
            nn.Flatten(), # the above ends up with batch_size x 512 x 1 x 1, flatten to batch_size x 512
        )
        
        self.harmonic_embedding_pos = HarmonicEmbedding(3, 2)
        self.pos_feature = 3 * 2 * 2 + 3
        self.feature_branch = nn.Sequential(
            nn.Linear(512+self.pos_feature, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

    def forward(self, x, desired_pos, return_embedding=False):
        with torch.no_grad():
            x_freq = freq_transform.dct_2d(x)
        # Concatenate x and x_freq along the channels
        x = torch.cat((x, x_freq), dim=1)
        feature = self.encoder(x) 
        feature_pos = self.harmonic_embedding_pos(desired_pos)
        feature_output = self.feature_branch(torch.cat((feature, feature_pos), dim=1))
        return feature_output
    

class ClassificationNetwork18_woDCT(nn.Module):
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
            SimpleResidualBlock(64, 128, 2),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 256, 2),
            SimpleResidualBlock(256, 256),
            SimpleResidualBlock(256, 512, 2),
            SimpleResidualBlock(512, 512), # 1024
            nn.AdaptiveAvgPool2d((1, 1)), # For each channel, collapses (averages) the entire feature map (height & width) to 1x1
            nn.Flatten(), # the above ends up with batch_size x 512 x 1 x 1, flatten to batch_size x 512
        )
        
        self.harmonic_embedding_pos = HarmonicEmbedding(3, 2)
        self.pos_feature = 3 * 2 * 2 + 3
        self.feature_branch = nn.Sequential(
            nn.Linear(512+self.pos_feature, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

    def forward(self, x, desired_pos, return_embedding=False):
        feature = self.encoder(x) 
        feature_pos = self.harmonic_embedding_pos(desired_pos)
        feature_output = self.feature_branch(torch.cat((feature, feature_pos), dim=1))
        return feature_output