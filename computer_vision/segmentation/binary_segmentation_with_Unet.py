import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors
import numpy as np
import os
import glob

"""
Loading the dataset
"""
transform_A = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.RandomHorizontalFlip(0.5),
    v2.ToDtype(torch.float32, scale=True)

])

class OxfordPetBinarySegmentation(datasets.OxfordIIITPet):
    def __init__(self, root,  split="trainval", image_transform=None, download=True):
        super().__init__(
            root=root,
            split=split,
            target_types="segmentation",
            download=download
        )
        self.image_transform = image_transform

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        # tell torchvision this is a segmentation mask
        mask = tv_tensors.Mask(np.array(mask))

        if self.image_transform:
            image, mask = self.image_transform(image, mask)

        # make binary mask
        binary_mask = (mask == 1).float()

        # ensure shape is [1,H, W]
        if binary_mask.ndim == 2:
            binary_mask = binary_mask.unsqueeze(0)

        return image, binary_mask

train_dataset = OxfordPetBinarySegmentation(
    root="/data",
    split="trainval",
    image_transform=transform_A,
    download=True
)

test_dataset = OxfordPetBinarySegmentation(
    root="/data",
    split="test",
    image_transform=transform_A,
    download=True
)

image, mask = train_dataset[0]
print(image.shape)
print(mask.shape)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False)

'''
model
'''

class Resblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
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
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.inc  = Resblock(in_ch, 64)     # 224
        self.enc1 = Resblock(64, 128)   # 112
        self.enc2 = Resblock(128, 256)  # 56
        self.enc3 = Resblock(256, 512)  # 28
        self.enc4 = Resblock(512, 1024) # 14

        # Decoder
        self.up1  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = Resblock(1024, 512)
        self.up2  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = Resblock(512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = Resblock(256, 128)
        self.up4  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = Resblock(128, 64)

        self.final_out  = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.pool(self.enc1(x1))
        x3 = self.pool(self.enc2(x2))
        x4 = self.pool(self.enc3(x3))
        x5 = self.pool(self.enc4(x4))

        # Decoder
        x = self.up1(x5)
        x = torch.concat([x, x4], dim=1)
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

        x = self.final_out(x)

        return x
     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Unet(in_ch=3, out_ch=1).to(device)
loss_fn = nn.BCEWithLogitsLoss()
lr = 3e-04
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 50

"""
Build the training and test loop
"""
def train(model, loader, loss_fn, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y.float())

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_train_loss = total_loss / total_samples

        print(f"[{epoch+1}/{epochs}] train_loss: {avg_train_loss:.4f}")

    return avg_train_loss


def dice_score(preds, targets, eps=1e-7):
    preds   = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union        = preds.sum(dim=1) + targets.sum(dim=1)

    dice         = (2 * intersection + eps) / (union + eps)

    return dice.mean()

def iou_score(preds, targets, eps=1e-7):
    preds   = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union        = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou          = (intersection + eps) / (union + eps)

    return iou.mean()


def test(model, loader, loss_fn, device, threshold=0.5):
    model.eval()

    total_loss    = 0.0
    total_samples = 0

    dice_total    = 0.0
    iou_total     = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss   = loss_fn(logits, y.float())

            probs  = torch.sigmoid(logits)
            preds  = (probs > threshold).float()

            batch_size = x.size(0)

            total_loss += loss.item() *  batch_size
            total_samples += batch_size

            dice_total = dice_score(preds, y).item() * batch_size
            iou_total  = iou_score(preds, y).item() * batch_size

    avg_loss = total_loss / total_samples
    avg_dice = dice_total / total_samples
    avg_iou  = iou_total / total_samples

    return avg_loss, avg_dice, avg_iou

if __name__ == "__main__":
    avg_train_loss = train(model, train_loader, loss_fn, optimizer, epochs, device)
    # avg_loss, avg_dice, avg_iou = test(model, test_loader, loss_fn, device, threshold=0.5)
    # print(f"Test phase\n")
    # print(f"avg_loss: {avg_loss}, avg_dice: {avg_dice}, avg_iou: {avg_iou}")
