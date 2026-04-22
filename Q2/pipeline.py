import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil



class CityScapesDataset(Dataset):
    def __init__(self, rgb_dir, mask_dir, transform=None, mask_transform=None):
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        rgb_path = os.path.join(self.rgb_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load RGB image
        image = Image.open(rgb_path).convert('RGB')
        # Load mask and force to single channel (grayscale / label IDs)
        mask = Image.open(mask_path).convert('L')   # <-- FIX: converts RGBA -> L (8-bit grayscale)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Now mask is a PIL Image in mode 'L' -> np.array gives (H, W)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask

# Define transforms
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
])

# Create full dataset
full_dataset = CityScapesDataset(
    rgb_dir='/home/m25csa012/CameraRGB',
    mask_dir='/home/m25csa012/CameraMask',
    transform=image_transform,
    mask_transform=mask_transform
)

# Split into train (80%) and test (20%) with seed 42
torch.manual_seed(42)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 8   # adjust based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=23, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Upsampling path
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(feature*2, feature))

        # Final classifier
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]

            # Handle possible size mismatch
            if x.shape != skip.shape:
                x = transforms.Resize(skip.shape[2:])(x)

            x = torch.cat((skip, x), dim=1)
            x = self.decoders[idx](x)

        return self.final_conv(x)

# Instantiate model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=23).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def compute_metrics(preds, targets, num_classes=23):
    """
    preds: (B, H, W) predicted class indices
    targets: (B, H, W) ground truth class indices
    Returns: mIoU (float), mDice (float)
    """
    iou_per_class = []
    dice_per_class = []
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union == 0:
            iou = float('nan')   # ignore if class absent in this batch
        else:
            iou = (intersection / union).item()
        iou_per_class.append(iou)

        # Dice = 2*intersection / (pred_c.sum() + target_c.sum())
        denom = pred_c.sum() + target_c.sum()
        if denom == 0:
            dice = float('nan')
        else:
            dice = (2 * intersection / denom).item()
        dice_per_class.append(dice)

    # Mean over classes, ignoring NaNs
    mIoU = np.nanmean(iou_per_class)
    mDice = np.nanmean(dice_per_class)
    return mIoU, mDice


num_epochs = 15
train_losses = []
test_mious = []
test_mdices = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)            # (B, 23, H, W)
        loss = criterion(outputs, masks)   # CrossEntropy expects (B, C, H, W) and (B, H, W)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)

    # ----- Evaluation on test set -----
    model.eval()
    all_mious = []
    all_mdices = []
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)   # (B, H, W)
            mIoU, mDice = compute_metrics(preds.cuda(), masks.cuda(), num_classes=23)
            if not np.isnan(mIoU):
                all_mious.append(mIoU)
            if not np.isnan(mDice):
                all_mdices.append(mDice)

    epoch_miou = np.mean(all_mious)
    epoch_mdice = np.mean(all_mdices)
    test_mious.append(epoch_miou)
    test_mdices.append(epoch_mdice)

    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Test mIoU = {epoch_miou:.4f}, Test mDice = {epoch_mdice:.4f}")

print("Training completed.")

torch.save(model.state_dict(), 'unet_cityscapes.pth')

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(test_mious, label='Test mIoU', color='green')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.title('Test mIoU')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(test_mdices, label='Test mDice', color='orange')
plt.xlabel('Epoch')
plt.ylabel('mDice')
plt.title('Test mDice')
plt.legend()

plt.tight_layout()
plt.savefig('training_plots.png')
plt.show()

