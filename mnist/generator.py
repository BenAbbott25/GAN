import torch.nn as nn

class Generator_conv(nn.Module):
    def __init__(self, channels: int):
        super(Generator_conv, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=channels * 2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.middle_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=channels * 2,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Sigmoid()
        )
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.middle_layer(x)
        x = self.downsample(x)
        x = self.final_layer(x)
        x = self.downsample(x)
        return x
    
class Generator_conv_simple(nn.Module):
    def __init__(self, channels: int):
        super(Generator_conv_simple, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
class Generator_linear(nn.Module):
    def __init__(self, input_size: int):
        super(Generator_linear, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.BatchNorm1d(input_size * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.middle_layer = nn.Sequential(
            nn.Linear(input_size * 2, input_size * 2),
            nn.BatchNorm1d(input_size * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_layer = nn.Sequential(
            nn.Linear(input_size * 2, input_size),
            nn.BatchNorm1d(input_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # convert from 28x28 to 1D
        x = x.view(x.size(0), -1)
        x = self.initial_layer(x)
        x = self.middle_layer(x)
        x = self.final_layer(x)
        return x.view(-1, 1, 28, 28)