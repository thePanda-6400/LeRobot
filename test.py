import torch
from model import CNN
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset import GraspDataset
from torch.utils.data import Subset



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = CNN().to(device)                          # move the model

# Load the weights from 'grasp_model.pt' to test the model
model.load_state_dict(torch.load('grasp_model.pt'))
model.eval()
