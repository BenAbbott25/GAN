import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_res: int, grayscale: bool = False):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1 if grayscale else 3, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(2 * input_res * input_res, 1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return self.activation(x)