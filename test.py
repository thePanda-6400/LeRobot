import torch
from model import CNN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

model = CNN().to(device)                          # move the model
fake = torch.randn(8, 3, 128, 128).to(device)     # move the input

out = model(fake)
print(out.shape)      # torch.Size([8, 3])
print(out.device)     # mps:0