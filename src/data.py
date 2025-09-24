# Data preprocessing and loading utilities for medical image classification
# This module handles the preparation of medical cancer images for training and evaluation
import os, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Standard image size for the model - all images will be resized to 256x256 pixels
# This ensures consistent input dimensions for the neural networks
IMG_SIZE = 256

def build_transforms(train=True):
    """
    Build image transformation pipelines for training and evaluation.
    
    Args:
        train (bool): If True, applies data augmentation for training.
                      If False, applies minimal transforms for validation/testing.
    
    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline
    """
    # Normalize pixel values to [-1, 1] range (standard for GANs)
    # Mean=0.5, std=0.5 maps [0,1] to [-1,1]: (x - 0.5) / 0.5 = 2x - 1
    norm = transforms.Normalize(mean=[0.5], std=[0.5])
    if train:
        # Training transforms include data augmentation to improve model generalization
        aug = transforms.Compose([
            # Convert to grayscale (medical images are often single-channel)
            transforms.Grayscale(num_output_channels=1),
            # Resize images to standard size for consistent model input
            transforms.Resize([IMG_SIZE, IMG_SIZE]),
            # Random rotation up to 10 degrees - simulates different scanning angles
            transforms.RandomRotation(degrees=10),
            # Random affine transformations - simulates patient positioning variations
            # translate: up to 5% shift in x,y directions
            # scale: random scaling between 90% and 110%
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9,1.1)),
            # Random horizontal flip - increases data diversity (50% probability)
            transforms.RandomHorizontalFlip(p=0.5),
            # Slight color variations - simulates different imaging conditions
            # Small brightness/contrast changes (5%) to improve robustness
            transforms.ColorJitter(brightness=0.05, contrast=0.05), 
            # Convert PIL Image to PyTorch tensor [0,1] range
            transforms.ToTensor(),
            # Apply normalization to [-1,1] range
            norm,
        ])

    else:
        # Validation/test transforms - no augmentation, only essential preprocessing
        aug = transforms.Compose([
            # Convert to grayscale for consistency with training
            transforms.Grayscale(num_output_channels=1),
            # Resize to standard size (no random cropping for reproducible results)
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # Convert to tensor
            transforms.ToTensor(),
            # Apply same normalization as training
            norm,
        ])

    return aug


def build_loaders(data_root, batch_size=32, num_workers=4):
    """
    Build PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        data_root (str): Root directory containing train/valid/test subdirectories
        batch_size (int): Number of images per batch (default: 32 for memory efficiency)
        num_workers (int): Number of parallel workers for data loading (default: 4)
    
    Returns:
        tuple: (train_dl, val_dl, test_dl, classes)
            - train_dl: DataLoader for training data with augmentation
            - val_dl: DataLoader for validation data without augmentation  
            - test_dl: DataLoader for test data without augmentation
            - classes: List of class names (e.g., cancer types)
    """
    # Create datasets using ImageFolder - expects directory structure:
    # data_root/
    #   ├── train/
    #   │   ├── class_0/
    #   │   ├── class_1/
    #   │   └── ...
    #   ├── valid/
    #   └── test/
    
    # Training dataset with augmentation transforms
    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=build_transforms(True))
    
    # Validation dataset without augmentation (for unbiased evaluation during training)
    val_ds = datasets.ImageFolder(os.path.join(data_root, "valid"), transform=build_transforms(False))
    
    # Test dataset without augmentation (for final model evaluation)
    test_ds = datasets.ImageFolder(os.path.join(data_root, "test"), transform=build_transforms(False))
    
    # Create DataLoaders for efficient batch processing
    # Training: shuffle=True for better learning, prevents overfitting to data order
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) # pin_memory=True speeds up GPU transfer by using pinned (page-locked) memory
    
    # Validation/Test: shuffle=False for reproducible evaluation results
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl, train_ds.classes