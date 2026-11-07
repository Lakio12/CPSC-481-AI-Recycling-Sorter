
"""
Preprocessing for the trashnet dataset

To Run:
python dataset.py

Output:
Classes: ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
Training images: 1768
Validation images: 379
Testing images: 380
Image batch shape: torch.Size([32, 3, 224, 224])
Label batch shape: torch.Size([32])
"""

# Imports
# =================================================
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


# Configurations
# =================================================
# Would need to change this if you wanna use this.
DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "dataset-resized"

# EfficientNet-B0 preprocessing

# Transforming the training data
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Transforming the Evaluation dataset
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Functions
# =================================================
# Getting dataloaders to use for the model
def get_dataloaders(batch_size=32):
    # Load dataset without applying transforms yet
    dataset = datasets.ImageFolder(DATA_DIR)

    labels = dataset.targets
    indices = list(range(len(dataset)))

    # 70% training, 30% remaining
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )

    # Split remaining data equally into validation and testing
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42,
        stratify=[labels[i] for i in temp_idx]
    )

    # Create datasets with different transforms
    train_dataset = datasets.ImageFolder(
        DATA_DIR, transform=train_transform
    )

    eval_dataset = datasets.ImageFolder(
        DATA_DIR, transform=eval_transform
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        Subset(eval_dataset, val_idx),
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        Subset(eval_dataset, test_idx),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, dataset.classes


# Main
# =================================================
# Test for the preprocessing

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    print("Classes:", classes)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Testing images:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
