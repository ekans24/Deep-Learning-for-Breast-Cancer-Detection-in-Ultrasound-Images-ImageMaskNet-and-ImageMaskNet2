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
from sklearn.metrics import precision_score, recall_score

# Define constants and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1
BATCH_SIZE = 8
IMAGE_SIZE = 256
NUM_CLASSES = 3

# Custom Model Architecture
class ImageMaskNet(nn.Module):
    def __init__(self):
        super(ImageMaskNet, self).__init__()
        # Image branch
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Mask branch
        self.mask_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Combined layers
        self.combined_fc = nn.Sequential(
            nn.Linear(64 * 2 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x_image, x_mask):
        x_image = self.image_branch(x_image)
        x_mask = self.mask_branch(x_mask)
        # Flatten and concatenate features from both branches
        x_image = x_image.view(x_image.size(0), -1)
        x_mask = x_mask.view(x_mask.size(0), -1)
        combined_features = torch.cat((x_image, x_mask), dim=1)
        output = self.combined_fc(combined_features)
        return output

class CustomImageMaskDataset(Dataset):
    def __init__(self, image_root_dir, mask_root_dir, image_transform=None):
        self.image_root_dir = image_root_dir
        self.mask_root_dir = mask_root_dir
        self.image_transform = image_transform
        self.image_paths, self.mask_paths, self.labels = self.get_image_mask_paths_and_labels()

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

        return image, mask, label, image_path  # Always return the image path

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

# Define image transformations
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
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
test_dataset = CustomImageMaskDataset(image_root_dir, mask_root_dir, image_transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Define lists to store metrics for plotting
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
train_precisions = []
test_precisions = []
train_recalls = []
test_recalls = []

# Function to compute precision and recall
def calculate_precision_recall(model, data_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, masks, labels, _ in data_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            outputs = model(images, masks)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predictions, average='weighted')
    return precision, recall

# Modify your train_model function to calculate precision and recall
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (images, masks, labels, _) in enumerate(train_loader, 0):  
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, masks)
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

        # Calculate precision and recall for training set
        train_precision, train_recall = calculate_precision_recall(model, train_loader)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)

        # Calculate precision and recall for testing set
        test_precision, test_recall = calculate_precision_recall(model, test_loader)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)

        # Append losses and accuracies for plotting
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Test the model and get the test loss for each epoch
        test_accuracy, _, _, avg_test_loss = test_model(model, test_loader, criterion)
        test_accuracies.append(test_accuracy)
        test_losses.append(avg_test_loss) 


    print("Finished Training")
    return model

def test_model(model, test_loader, criterion):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    test_loss = 0  # Initialize test loss

    with torch.no_grad():
        for images, masks, labels, _ in test_loader:  
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            outputs = model(images, masks)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            # Calculate and accumulate the test loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    test_accuracy = correct_predictions / total_predictions
    avg_test_loss = test_loss / len(test_loader)  # Calculate average test loss
    print(f"Test Accuracy: {test_accuracy:.4f} - Test Loss: {avg_test_loss:.4f}")

    # Calculate precision and recall for testing set
    test_precision, test_recall = calculate_precision_recall(model, test_loader)
    return test_accuracy, test_precision, test_recall, avg_test_loss  # Return test loss


# Train the model
model = train_model(model, train_loader, test_loader, criterion, optimizer)

# Test the model and get precision, recall, and test loss
test_accuracy, test_precision, test_recall, avg_test_loss = test_model(model, test_loader, criterion)


# Print the final precision and recall scores
print(f"Final Test Precision: {test_precision:.4f}")
print(f"Final Test Recall: {test_recall:.4f}")

# Visualization function
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Visualization function
def visualize_original_images(model, class_names, data_loader, num_images_per_class=3):
    was_training = model.training
    model.eval()
    fig = plt.figure(figsize=(15, 10))

    class_images_count = {classname: 0 for classname in class_names}

    with torch.no_grad():
        for images, masks, labels, paths in data_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)

            outputs = model(images, masks)
            _, preds = torch.max(outputs, 1)

            for j in range(len(labels)):
                class_label = class_names[labels[j]]
                predicted_label = class_names[preds[j]]

                if class_images_count[class_label] < num_images_per_class:
                    class_images_count[class_label] += 1

                    ax = plt.subplot(num_images_per_class, len(class_names), sum(class_images_count.values()))
                    ax.axis('off')
                    ax.set_title(f'True: {class_label}, Predicted: {predicted_label}')

                    # Load and display the original image from the path
                    original_image = Image.open(paths[j])
                    plt.imshow(original_image)

                if all(count == num_images_per_class for count in class_images_count.values()):
                    model.train(mode=was_training)
                    plt.show()
                    return
        model.train(mode=was_training)
        plt.show()

# Define class names
class_names = ['benign', 'malignant', 'normal']

# Visualize original images with true and predicted labels
visualize_original_images(model, class_names, test_loader)

# Plot loss vs. epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epoch')
plt.grid(True)
plt.show()

# Plot accuracy vs. epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, NUM_EPOCHS + 1), test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epoch')
plt.grid(True)
plt.show()