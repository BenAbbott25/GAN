import torch.nn as nn

class Discriminator_conv(nn.Module):
    def __init__(self, channels: int):
        super(Discriminator_conv, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels * 2,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 2,
                out_channels=channels * 4,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 4,
                out_channels=channels * 8,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flatten = nn.Flatten()
        self.dense_layer = nn.Sequential(
            # nn.Linear(channels * 8 * 4 * 4, 256),
            nn.Linear(196, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.dense_layer(x)
    
class Discriminator_linear(nn.Module):
    def __init__(self, input_size: int):
        super(Discriminator_linear, self).__init__()
        self.dense_layer = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.BatchNorm1d(input_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_size * 2, input_size * 2),
            nn.BatchNorm1d(input_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.dense_layer(x)