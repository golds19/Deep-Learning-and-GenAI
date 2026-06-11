import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision import tv_tensors
from torchvision.transforms import v2
import os 
import glob
import numpy as np
from tqdm import tqdm


transform_A = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor()
])

class OxfordBinaryDataset(datasets.OxfordIIITPet):
    def __init__(self, root, split, target_types, download, transform=None):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
        )

        self.transform = transform

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        # convert to tv tensor image
        image_tensor = v2.functional.to_image(image)
        # convert mask to mask
        mask = tv_tensors.Mask(np.array(mask))

        if self.transform:
            image_tensor, mask = self.transform(image, mask)

        # make binary mask
        binary_mask = (mask == 1).float()

        # ensure binary mask shape is (1,H,W)
        if binary_mask.ndim == 2:
            binary_mask = binary_mask.unsqueeze(0)

        return image_tensor, binary_mask
    
train_dataset = OxfordBinaryDataset(
    root="/data",
    split="trainval",
    target_types="segmentation",
    download=True,
    transform=transform_A
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

'''
Building the model
'''
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
    

class ResUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.pool = nn.AvgPool2d(2,2)

        # encoder
        self.inc  = Resblock(in_ch, 64)
        self.enc1 = Resblock(64, 128)
        self.enc2 = Resblock(128, 256)
        self.enc3 = Resblock(256, 512)
        self.enc4 = Resblock(512, 1024)

        # decoder
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
        # encoder
        x1 = self.inc(x)
        x2 = self.enc1(self.pool(x1))
        x3 = self.enc2(self.pool(x2))
        x4 = self.enc3(self.pool(x3))
        x5 = self.enc4(self.pool(x4))

        # decoder
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec1(x)

        x = self.up2(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)

        x = self.up3(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.dec3(x)

        x = self.up4(x2)
        x = torch.cat([x, x1], dim=1)
        x = self.dec4(x)

        x = self.final_conv(x)

        return x

def training_loop(model, dataloader, loss_fn, optimizer, epochs, device):
    for epoch in tqdm(range(epochs)):
        model.train() # put the model in train mode
        train_loss    = 0.0
        total_samples = 0

        for batch, mask in dataloader:
            batch, mask = batch.to(device), mask.to(device)
            optimizer.zero_grad()

            logits = model(batch)
            loss   = loss_fn(logits, mask.float())

            loss.backward()
            optimizer.step()

            batch_size = batch.size(0)
            train_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_train_loss = train_loss / total_samples

        print(f"[{epoch+1}/{epochs}] training_loss: {avg_train_loss:.4f}")

    return avg_train_loss

if __name__ == "__main__":
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = ResUnet(in_ch=3, out_ch=1).to(device)
    lr = 1e-04
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn   = nn.BCEWithLogitsLoss()
    epochs    = 50

    avg_train_loss = training_loop(
        model,
        train_loader,
        loss_fn,
        optimizer,
        epochs,
        device
    )
    