# Import necessary libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define constants and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 5
BATCH_SIZE = 16
IMAGE_SIZE = 256
NUM_CLASSES = 3

class ImageMaskNet(nn.Module):
    def __init__(self):
        super(ImageMaskNet, self).__init__()
        # Image branch with increased depth and batch normalization
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Mask branch with increased depth
        self.mask_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        
        # Dummy forward pass to determine size of the flattened features
        with torch.no_grad():
            self.image_feature_size = None
            self.combined_feature_size = None
            self.dummy_forward(torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE)), 
                               torch.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE)))
        # Combined layers
        self.fc_image_only = nn.Sequential(
            nn.Linear(self.image_feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CLASSES)
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(self.combined_feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CLASSES)
        )

    def dummy_forward(self, x_image, x_mask):
        x_image = self.image_branch(x_image)
        x_mask = self.mask_branch(x_mask)
        x_image = x_image.view(x_image.size(0), -1)
        x_mask = x_mask.view(x_mask.size(0), -1)
        combined_features = torch.cat((x_image, x_mask), dim=1)
        if self.image_feature_size is None:
            self.image_feature_size = x_image.shape[1]
        if self.combined_feature_size is None:
            self.combined_feature_size = combined_features.shape[1]

    def forward(self, x_image, x_mask=None):
        x_image = self.image_branch(x_image)
        if x_mask is not None:
            x_mask = self.mask_branch(x_mask)
            x_mask = x_mask.view(x_mask.size(0), -1)
            x_image = x_image.view(x_image.size(0), -1)
            combined_features = torch.cat((x_image, x_mask), dim=1)
            output = self.fc_combined(combined_features)
        else:
            x_image = x_image.view(x_image.size(0), -1)
            output = self.fc_image_only(x_image)
        return output



class CustomImageMaskDataset(Dataset):
    def __init__(self, image_root_dir, mask_root_dir, image_transform=None, use_masks=True):
        self.image_root_dir = image_root_dir
        self.mask_root_dir = mask_root_dir
        self.image_transform = image_transform
        self.image_paths, self.mask_paths, self.labels = self.get_image_mask_paths_and_labels()
        self.use_masks = use_masks

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale

        if self.image_transform:
            image = self.image_transform(image)
            mask = self.image_transform(mask)
        
        if not self.use_masks:
            mask = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)  # A placeholder for no mask case
        else:
            mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale
            if self.image_transform:
                mask = self.image_transform(mask)

        return image, mask, label, image_path

    def get_image_mask_paths_and_labels(self):
        image_paths = []
        mask_paths = []
        labels = []
        for category in ['benign', 'malignant', 'normal']:
            images_dir = os.path.join(self.image_root_dir, category, 'images')
            masks_dir = os.path.join(self.mask_root_dir, category, 'masks')
            image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
            mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('_mask.png')])

            image_paths.extend([os.path.join(images_dir, fname) for fname in image_files])
            mask_paths.extend([os.path.join(masks_dir, fname) for fname in mask_files])
            labels.extend([category] * len(image_files))

        label_to_index = {'benign': 0, 'malignant': 1, 'normal': 2}
        labels = [label_to_index[label] for label in labels]

        return image_paths, mask_paths, labels


# Set the root directory for images and masks
image_root_dir = 'Dataset_BUSI_with_GT'
mask_root_dir = 'Dataset_BUSI_with_GT'

# Define image transformations with additional data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Add random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add color jitter
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Instantiate the model
model = ImageMaskNet().to(device)

# Dataset and DataLoader
train_dataset = CustomImageMaskDataset(image_root_dir, mask_root_dir, image_transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = CustomImageMaskDataset(image_root_dir, mask_root_dir, image_transform=test_transforms, use_masks=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Update the train_model function to include the scheduler
def train_model(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    model.train()
    half_epoch = num_epochs // 2
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (images, masks, labels, _) in enumerate(train_loader, 0):
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if epoch < half_epoch:
                outputs = model(images, masks)
            else:
                outputs = model(images)  # Use only images in the second phase

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    print("Finished Training")
    return model

# Train the model with the scheduler
model = train_model(model, train_loader, criterion, optimizer)


def test_model(model, test_loader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, _, labels, _ in test_loader:  # Ignore masks
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    print(f"Test Accuracy: {correct_predictions / total_predictions:.4f}")

# Test the model
test_model(model, test_loader)
