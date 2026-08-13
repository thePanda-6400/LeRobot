import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from model import CNN
from dataset import GraspDataset

OVERFIT_TEST = False

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

dataset = GraspDataset()

def criterion(x, y):
    return ((x - y) ** 2).mean()

model = CNN().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

if OVERFIT_TEST:
    idxs = np.linspace(0, len(dataset)-1, 10).astype(int).tolist()
    tiny = Subset(dataset, idxs)
    loader = DataLoader(tiny, batch_size=10, shuffle=True)

    for epoch in range(300):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()


        if epoch % 20 == 0:
            print(f"epoch {epoch:3d} | loss {loss.item():.6f}")

    print(f"final overfit loss: {loss.item():.6f}")
    print("PASS if this is near zero (~0.0001 or less)")

else:
    n_test = int(0.2 * len(dataset))
    n_train = len(dataset) - n_test
    gen = torch.Generator().manual_seed(42)
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=gen)
    print(f"train: {len(train_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    def eval_loss(loader):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                total += criterion(model(inputs), labels).item()
        model.train()
        return total / len(loader)

    for epoch in range(200):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == 49:
            test_loss = eval_loss(test_loader)
            print(f"epoch {epoch:3d} | train {train_loss:.6f} | test {test_loss:.6f}")

    torch.save(model.state_dict(), "grasp_model.pt")
    print("saved grasp_model.pt")