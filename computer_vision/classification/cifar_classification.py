import torch
import torch.nn as nn
import  torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

'''
Load the dataset
'''
train_dataset = datasets.CIFAR10(
    root="/data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
val_dataset = datasets.CIFAR10(
    root="/data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# making the dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=8,shuffle=True)
val_loader   = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)


'''
building the model
'''

class TestModel(nn.Module):
    def __init__(self, in_ch, hidden):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, stride=1),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden, 128, kernel_size=3, stride=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.maxPool = nn.MaxPool2d(2)
        self.linear  = nn.LazyLinear(3)

    def forward(self, x):
        x = self.maxPool(self.conv1(x))
        x = self.maxPool(self.conv2(x))

        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}d")

model  = TestModel(in_ch=3, hidden=64).to(device)
epochs = 50
lr = 1e-03
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

'''
training loop
'''

def training_loop(model, train_loader, loss_fn, optimizer, device):
    '''
    model on training model
    '''
    model.train()
    train_loss, train_avg_loss = 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_avg_loss = train_loss / len(train_loader)

    return train_loss, train_avg_loss


def val_loop(model, val_loader, loss_fn, device):
    '''
    model eval
    '''
    model.eval(0)
    val_loss, val_avg_loss = 0, 0

    with torch.no_grad:
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            val_loss+=loss.item()

        val_avg_loss = val_loss / len(val_loader)

        return val_loss, val_avg_loss
    

def train(model, epochs, train_loader, val_loader, loss_fn, optimizer, device):
    
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        _, train_avg_loss = training_loop(model, train_loader, loss_fn, optimizer, device)
        _, val_avg_loss     = val_loop(model, val_loader, loss_fn, device)

        train_losses.append(train_avg_loss)
        val_losses.append(val_avg_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses:.2f}, Validation Loss: {val_losses:.2f}")
              
        history = {
            "train_losses": train_losses,
            "val_losses": val_losses
        }

        return history


if __name__ == "__main__":
    history = train(
        model,
        epochs,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device
    )