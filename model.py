import torch
import torch.nn as nn
import torch.nn.functional as F


def add_coords(x):
    """Append two channels encoding each pixel's x and y position.
    (batch, C, H, W) -> (batch, C+2, H, W)"""
    b, c, h, w = x.shape
    xs = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
    ys = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
    return torch.cat([x, xs, ys], dim=1)


class CNN(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), num_outputs=3):
        super().__init__()
        in_ch = input_shape[0] + 2               # +2 for the coordinate channels
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = add_coords(dummy)            # <-- coords added before convs
            n_flat = self.flatten(self.pool(self.conv3(self.conv2(self.conv1(dummy))))).shape[1]
        print("flattened size:", n_flat)

        self.linear1 = nn.Linear(n_flat, 128)
        self.linear2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = add_coords(x)                        # <-- the key line
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        return self.linear2(x)