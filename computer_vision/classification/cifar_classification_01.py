import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as Basedataset
from torchvision.transforms import transforms
import numpy as np
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
batch_size = 8
epochs = 10
loss_fn = nn.CrossEntropyLoss()



"""
Loading the dataset from datasets
"""
train_dataset = datasets.CIFAR10(
    root="/data",
    train=True,
    donload=True,
    transform=transforms.ToTensor()
)
val_dataset = datasets.CIFAR10(
    root="/data",
    train=False,
    download=True,
    transform = transforms.ToTensor()
)

'''
making the dataloaders
'''
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

'''
make the model
'''

class Model01(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(2)
        self.linear  = nn.LazyLinear(3) # expected outputs

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = torch.flatten(x, 1) # flattten from dim=1
        x = self.linear(x)

        return x
    

'''
making the training loop
'''

def training_loop(model, train_loader, loss_fn, optimizer, device):
    model.train()
    train_loss, avg_train_loss = 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss   = loss_fn(logits, y)

        loss+=loss.item()

    avg_train_loss = train_loss / len(train_loader)

    return avg_train_loss

def eval_loop(model, val_loader, loss_fn, device):
    model.eval()
    eval_loss, avg_eval_loss = 0,0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = loss_fn(logits, y)

        eval_loss+=loss.item()

    avg_eval_loss = eval_loss / len(val_loader)

    return avg_eval_loss

def train(model, train_loader, val_loader, loss_fn, optimizer, device):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        avg_train_loss = training_loop(model, train_loader, loss_fn, optimizer, device)
        avg_eval_loss  = eval_loop(model, val_loader, loss_fn, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_eval_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses:.2f}, Validation Loss: {val_losses:.2f}")

        history = {
            "train_losses": train_losses,
            "val_losses": val_losses
        }

        return history
        

if __name__ == "__main__":
    x, y  = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    model = Model01(in_ch=3).to(device)
    out   = model(x)

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"out: {out.shape}")
