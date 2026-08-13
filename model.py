
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), num_outputs=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride = 2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride = 2, padding=1)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            n_flat = self.flatten(self.conv2(self.conv1(dummy))).shape[1]

        self.linear1 = nn.Linear(n_flat, num_outputs)

    


    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        return self.linear1(x)
    

