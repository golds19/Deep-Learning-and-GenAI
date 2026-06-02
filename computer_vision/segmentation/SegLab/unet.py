import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

'''
loading the dataset
using torchvision dataset
'''
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.OxfordIIITPet(
    root="/data",
    download=True,
    split="trainval",
    target_types="category",
    transform=train_transform
)

test_dataset = datasets.OxfordIIITPet(
    root='/data',
    split="test",
    download=True,
    target_types="category",
    transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Resblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch)
        )

        # shortcut path
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv_block(x)
        x = x + identity
        return self.relu(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)    


class ResUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        self.pool = nn.MaxPool2d(2,2)

        # Encoder
        self.inc  = Resblock(3, 64)
        self.enc1 = Resblock(64, 128)
        self.enc2 = Resblock(128, 256)
        self.enc3 = Resblock(256, 512)
        
        self.enc4 = Resblock(512, 1024) # bottleneck


        # Decoder
        '''
        1. upsample
        2. apply doublwconv
        '''
        self.up1  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = Resblock(1024, 512)

        self.up2  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = Resblock(512, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = Resblock(256, 128)

        self.up4  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = Resblock(128, 64)

        self.final_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x) # 320
        x2 = self.enc1(self.pool(x1)) # 160
        x3 = self.enc2(self.pool(x2)) # 80
        x4 = self.enc3(self.pool(x3)) # 40
        x5 = self.enc4(self.pool(x4)) # 20


        # Decoder
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec1(x)

        x = self.up2(x4)
        x = torch.concat([x, x3], dim=1)
        x = self.dec2(x)

        x = self.up3(x3)
        x = torch.concat([x, x2], dim=1)
        x = self.dec3(x)

        x = self.up4(x2)
        x = torch.concat([x, x1], dim=1)
        x = self.dec4(x)

        return self.final_conv(x)
    
'''
train loop
'''

def train_loop(model, train_loader, loss_fn, optimizer, device):
    '''
    putting the model in train mode
    '''
    model.train()
    avg_train_loss, train_loss = 0,0
    for x, y in train_loader:
        x, y   = x.to(device), y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss   = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    return avg_train_loss


def test_loop(model, test_loader, loss_fn, device):
    '''
    putting the model on eval
    '''
    model.eval()
    avg_test_loss, test_loss =  0,0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = loss_fn(logits, y)

            test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)

        return avg_test_loss

def train(model, train_loader, test_loader, loss_fn, optimizer, epochs, device):

    train_losses, test_losses = [], []
    
    for epoch in range(epochs):
        avg_train_loss = train_loop(model, train_loader, loss_fn, optimizer, device)
        avg_test_loss  = test_loop(model, test_loader, loss_fn, device)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses:.2f}, Validation Loss: {test_losses:.2f}")

        history = {
            "train_losses": train_losses,
            "val_losses": test_losses
        }

        return history
    

if __name__ == "__main__":

    model = ResUNet(in_ch=3).to(device)

    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    
    out = model(x)
    print(out.shape)
