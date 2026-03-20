import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes=34):
        super().__init__()

        self.conv_layer = nn.Sequential(
            self._block_conv(3, 16),
            self._block_conv(16, 32),
            self._block_conv(32, 64),
            self._block_conv(64, 128),
            self._block_conv(128, 256)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.fc_layer = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256 * 3 * 3, 256),
            nn.LeakyReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )


    def _block_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2)
        )



    def forward(self, x):
        x = self.conv_layer(x)
        x = self.avgpool(x)
        x = self.fc_layer(x)

        return x
