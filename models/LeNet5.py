import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flattened_size = self._get_conv_output_size()

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 120), nn.ReLU(inplace=True),
            nn.Linear(120, 84), nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def _get_conv_output_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            x = self.features(x)
            return x.numel()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) 
        x = self.classifier(x)
        return x