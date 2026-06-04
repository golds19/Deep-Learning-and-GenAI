import torch
import torch.nn as nn
from torch.utils.data import random_split
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

'''
Loading the MNIST dataset
'''
dataset = datasets.MNIST(
    root='/data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

val_size   = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# creating the dataloaders
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=4, shuffle=False)


'''
building the model
'''

class Model(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.pool = nn.MaxPool2d(2)
        self.out  = nn.LazyLinear(10)

    def forward(self, x):
        x = self.conv(x)
        x = self.out(x)

        return x
    
'''
writing the training and test loop
'''
model = Model(in_ch=1)
epochs  = 50
loss_fn = nn.CrossEntropyLoss()
lr = 1e-03
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(model, train_loader, loss_fn, optimizer, device):
    # model on train mode
    model.train()
    avg_train_loss, train_loss = 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    return avg_train_loss

def val_loop(model, val_loader, loss_fn, device):
    # model on eval
    model.eval()
    avg_val_loss, val_loss = 0,0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        val_loss+= loss.item()

    avg_val_loss = val_loss / len(val_loader)

    return avg_val_loss


def train(model, train_loader, val_loader, loss_fn, optimizer, epochs, device):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        avg_train_loss = train_loop(model, train_loader, loss_fn, optimizer, device)
        avg_val_loss   = val_loop(model, val_loader, loss_fn, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Training loss: {avg_train_loss:.2f}, Validation loss: {avg_val_loss:.2f}")

        history = {
            "train_losses": train_losses,
            "val_losses": val_losses
        }

    return history


if __name__ == "__main__":
    training_dataset   = train_set
    validation_dataset = val_set

    train_loader = train_loader
    val_loader   = val_loader

    model = model.to(device)

    history = train(model, train_loader, val_loader, loss_fn, optimizer, epochs, device)