import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


"""
CartPole network
"""


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNN, self).__init__()

        # CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate the size of flattened features
        self.conv_output_size = self._get_conv_output_size(input_shape)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_output_size(self, shape):
        x = torch.zeros(1, *shape)
        x = x.permute(0, 3, 1, 2)  # Change to NCHW format
        x = self.conv(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        # Change input from NHWC to NCHW format
        if len(x.size()) == 4:
            x = x.permute(0, 3, 1, 2)
        elif len(x.size()) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
