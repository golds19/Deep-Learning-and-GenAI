"""
This script demonstrates how to train
a binary segmentation model using
Camvid dataset and segmentation_models_pytorch
"""

# import the libraries
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm

import segmentation_models_pytorch as smp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d:%m:%Y %H:%M:%S",
)

'''
set the device to gpu
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

'''
download the CamVid dataset
'''
os.makedirs("binary_segmentation_data", exist_ok=True)
main_dir = "binary_segmentation_data"

data_dir = os.path.join(main_dir, "dataset")
if not os.path.exists(data_dir):
    logging.info("Loading data...")
    os.system(f"git clone https://github.com/alexgkendall/SegNet-Tutorial {data_dir}")
    logging.info("Done!")

# Create a directory to store the output mask
output_dir = os.path.join(main_dir, "output_images")
os.makedirs(output_dir, exist_ok=True)

'''
Define the hyperparameters
'''
epochs_max = 200
adam_lr    = 2e-4
eta_min    = 1e-5
batch_size = 8
input_image_reshape = (320, 320)
foreground_class = 1

'''
Define a custom dataset
'''
class Dataset(BaseDataset):
    """
    A custom dataset class for binary segmentation task
    """

    def __init__(self, images_dir, masks_dir, input_image_reshape=(320,320), foreground_class=1, augmentation=None):
        
        self.ids = os.listdir(images_dir)
        self.images_filepath = [
            os.path.join(images_dir, image_id) for image_id in self.ids
        ]
        self.masks_filepath = [
            os.path.join(masks_dir, image_id) for image_id in self.ids
        ]

        self.input_image_reshape = input_image_reshape
        self.foreground_class    = foreground_class
        self.augmentation        = augmentation

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        '''
        retrieves the image and corresponding mask at index `i`
        '''
        # read the image
        image = cv2.imread(
            self.images_filepath[index], cv2.IMREAD_GRAYSCALE
        ) # read trhe image in grayscale
        image = np.expand_dims(image, axis=-1) # Add channel dimension (1,1)

        # resize the image
        image = cv2.resize(image, self.input_image_reshape)

        # read the mask in grayscale mode
        mask = cv2.imread(self.masks_filepath[index], cv2.IMREAD_GRAYSCALE)

        # Update the mask: Set foreground_class to 1 and the rest to 0
        mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)

        # resize the mask to input_image_reshape
        mask_remap = cv2.resize(mask_remap, self.input_image_reshape)

        if self.augmentation:
            sample            = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        # Convert to pytorch tensors
        # add channel dimension if missing
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # HWC -> CWH and normalize to [0, 1]
        image = torch.tensor(image).float().permute(2,0,1) / 255.0

        # ensure mask is LongTensor
        mask_remap = torch.tensor(mask_remap).long()

        return image, mask_remap

# define the model to be used
class CamVidModel(nn.Module):
    def __init__(self, arch, encoder_name, in_channels=3, out_classes=1, **kwargs):
        super().__init__()
        self.mean  = torch.tensor([0.485, 0.456, 0.496]).view(1,3,1,1).to(device)
        self.std   = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs
        )

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask  = self.model(image)
        return mask
        
def visualize(output_dir, image_filename, **images):
    """Plot images in one row"""
    n = len(images)
    plt.figure(figsize=(16,5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{name.split("-")}".title())
        plt.imshow(image)
    plt.show()
    plt.savefig(os.path.join(output_dir, image_filename))
    plt.close()


# use multiple CPUs in parallel
def train_and_evaluate_one_epoch(model, train_loader, valid_loader, optimizer, scheduler, losss_fn, device):

    # Set the model to training mode
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        images, masks = batch
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = losss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_loader)

    # Set the model on eval
    model.eval()
    val_loss = 0
    with torch.inference_mode():
        for batch in tqdm(valid_loader, desc="Evaluating"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss    = losss_fn(outputs, masks)

            val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)

        return avg_train_loss, avg_val_loss


def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, epochs):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        avg_train_loss, avg_valid_loss = train_and_evaluate_one_epoch(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_fn,
            device
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_valid_loss)

        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.2f}, Validation Loss: {avg_valid_loss:.2f}")

        history = {
            "train_losses": train_losses,
            "val_losses": val_losses
        }

        return history
    
def test_model(model, output_dir, test_dataloader, loss_fn, device):
    # Set the model to evaluation mode
    model.eval()
    test_loss = 0
    tp, fp, fn, tn = 0,0,0,0
    with torch.inference_mode():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            for i, output in enumerate(outputs):
                input  = images[i].cpu().numpy().transpose(1,2,0)
                output = output.squeeze().cpu().numpy()

                visualize(
                    output_dir,
                    f"output_{i}.png",
                    input_images=input, 
                    output_mask=output,
                    binary_mask =output > 0.5,
                )

            test_loss += loss.item()

            prob_mask = outputs.sigmoid().squeeze(1)
            pred_mask = (prob_mask > 0.5).long()
            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                pred_mask, masks, mode="binary"
            )
            tp += batch_tp.sum().item()
            fp += batch_tp.sum().item()
            fn += batch_fn.sum().item()
            tn += batch_tn.sum().item()

        test_loss_mean = test_loss / len(test_dataloader)
        logging.info(f"Test Loss: {test_loss_mean:.2f}")

    iou_score = smp.metrics.iou_score(
        torch.tensor([tp]),
        torch.tensor([fp]),
        torch.tensor([fn]),
        torch.tensor([tn]),
        reduction="micro"
    )

    return test_loss_mean, iou_score.item()

'''
Define the data directories and create the datasets
'''
x_train_dir = os.path.join(data_dir, "CamVid", "train")
y_train_dir = os.path.join(data_dir, "CamVid", "trainannot")

x_val_dir   = os.path.join(data_dir, "CamVid", "val")
y_val_dir   = os.path.join(data_dir, "CamVid", "valannot")

x_test_dir  = os.path.join(data_dir, "CamVid", "test")
y_test_dir  = os.path.join(data_dir, "CamVid", "testannot")

train_dataset = Dataset(x_train_dir, y_train_dir, input_image_reshape=input_image_reshape, foreground_class=foreground_class)
valid_dataset = Dataset(x_val_dir, y_val_dir, input_image_reshape=input_image_reshape, foreground_class=foreground_class)
test_dataset  = Dataset(x_test_dir, y_test_dir, input_image_reshape=input_image_reshape, foreground_class=foreground_class)

image, mask = train_dataset[0]
logging.info(f"Unique values in mask: {np.unique(mask)}")
logging.info(f"Image shape: {image.shape}")
logging.info(f"Mask shape: {mask.shape}")

# create the dataloaders
logging.info(f"Train size: {len(train_dataset)}")
logging.info(f"Validation size: {len(valid_dataset)}")
logging.info(f"Test size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Looking at some samples
sample = train_dataset[0]
visualize(
    output_dir,
    "train_sample.png",
    train_image=sample[0].numpy().transpose(1, 2, 0),
    train_mask=sample[1].squeeze(),
)

# Visualize and save validation sample
sample = valid_dataset[0]
visualize(
    output_dir,
    "validation_sample.png",
    validation_image=sample[0].numpy().transpose(1, 2, 0),
    validation_mask=sample[1].squeeze(),
)

# Visualize and save test sample
sample = test_dataset[0]
visualize(
    output_dir,
    "test_sample.png",
    test_image=sample[0].numpy().transpose(1, 2, 0),
    test_mask=sample[1].squeeze(),
)

# ----------------------------
# Create and train the model
# ----------------------------
max_iter = epochs_max * len(train_loader)  # Total number of iterations

model = CamVidModel("Unet", "resnet34", in_channels=3, out_classes=1)

# Training loop
model = model.to(device)

# Define the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

# Define the learning rate scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=eta_min)

# Define the loss function
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

# Train the model
history = train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs_max,
)

# Visualize the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(history["train_losses"], label="Train Loss")
plt.plot(history["val_losses"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.savefig(os.path.join(output_dir, "train_val_losses.png"))
plt.close()


# Evaluate the model
test_loss = test_model(model, output_dir, test_loader, loss_fn, device)

logging.info(f"Test Loss: {test_loss[0]}, IoU Score: {test_loss[1]}")
logging.info(f"The output masks are saved in {output_dir}.")