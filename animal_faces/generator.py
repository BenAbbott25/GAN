import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Generator2(nn.Module):
    def __init__(self, input_res: int, output_res: int, output_channels: int, greyscale: bool):
        super(Generator2, self).__init__()
        self.input_res = input_res
        print(f'input_res: {input_res}')
        self.conv1 = nn.Conv2d(in_channels=input_res ** 2, out_channels=(input_res // 2) ** 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=(input_res // 2) ** 2, out_channels=(input_res // 4) ** 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=(input_res // 4) ** 2, out_channels=(input_res // 8) ** 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=(input_res // 8) ** 2, out_channels=(input_res // 16) ** 2, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=(input_res // 16) ** 2, out_channels=(input_res // 32) ** 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = nn.Conv2d(in_channels=(input_res // 32) ** 2, out_channels=(input_res // 64) ** 2, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        # print(f'raw x: {x.shape}')
        x = x.view(x.size(0), self.input_res * self.input_res, 1, 1)  # Reshape input to (batch_size, 4096, 1, 1)
        # print(f'reshaped x: {x.shape}')
        x = self.upsample(x)
        # print(f'upsampled x: {x.shape}')
        x = self.activation(self.conv1(x))
        # print(f'conv1 x: {x.shape}')
        x = self.upsample(x)
        # print(f'upsampled x: {x.shape}')
        x = self.activation(self.conv2(x))
        # print(f'conv2 x: {x.shape}')
        x = self.upsample(x)
        # print(f'upsampled x: {x.shape}')
        x = self.activation(self.conv3(x))
        # print(f'conv3 x: {x.shape}')
        x = self.upsample(x)
        # print(f'upsampled x: {x.shape}')
        x = self.activation(self.conv4(x))
        # print(f'conv4 x: {x.shape}')
        x = self.upsample(x)
        # print(f'upsampled x: {x.shape}')
        x = self.activation(self.conv5(x))
        # print(f'conv5 x: {x.shape}')
        x = self.upsample(x)
        # print(f'upsampled x: {x.shape}')
        x = self.activation(self.conv6(x))
        # print(f'conv6 x: {x.shape}')
        return x
    
    def view_layer(self, x):
        print(f'layer: {x[0].shape}')
        layout = int((x[0].shape[0])**0.5)

        fig, axs = plt.subplots(layout, layout)
        if layout > 1:
            for i in range(layout):
                for j in range(layout):
                    axs[i, j].imshow(x[0][i * layout + j].detach().cpu().numpy())
        plt.show()
        plt.pause(0.1)


class Generator(nn.Module):
    def __init__(self, input_res: int, output_res: int, output_channels: int, greyscale: bool):
        super(Generator, self).__init__()
        self.input_res = input_res
        self.output_res = output_res
        self.output_channels = output_channels
        self.greyscale = greyscale

        self.conv1 = nn.Conv2d(in_channels=input_res ** 2, out_channels=(input_res // 4) ** 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=(input_res // 4) ** 2, out_channels=(input_res // 16) ** 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=(input_res // 16) ** 2, out_channels=(input_res // 64) ** 2, kernel_size=3, stride=1, padding=1)

        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')

        self.activation = nn.Sigmoid()  
        
    def forward(self, x):
        x = x.view(x.size(0), self.input_res * self.input_res, 1, 1)
        x = self.upsample(x)
        x = self.activation(self.conv1(x))
        x = self.upsample(x)
        x = self.activation(self.conv2(x))
        x = self.upsample(x)
        x = self.activation(self.conv3(x))
        return x