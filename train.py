import torch
from model import CNN
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset import GraspDataset
from torch.utils.data import Subset


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

dataset = GraspDataset()
tiny = Subset(dataset, range(10))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = CNN().to(device)

def criterion(x, y):
    return ((x - y) ** 2).mean()      # <-- .mean() makes it a scalar

optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(500):
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)   # <-- move to device

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if epoch%20 == 0:
        print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')