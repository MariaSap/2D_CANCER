import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np

IMG_SIZE = 256

def count_classes(dataroot):
    """Count the number of images per class in a folder and print the results."""
    class_counts = {}
    for clsname in sorted(os.listdir(dataroot)):
        clspath = os.path.join(dataroot, clsname)
        if os.path.isdir(clspath):
            num_imgs = sum(1 for fname in os.listdir(clspath) 
                          if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')))
            class_counts[clsname] = num_imgs
    
    if class_counts:
        print(f"\nClass counts in {dataroot}:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")
        min_count = min(class_counts.values())
        minority = [cls for cls, cnt in class_counts.items() if cnt == min_count]
        print(f"  Minority classes: {', '.join(minority)} with {min_count} instances each.")
    
    return class_counts

def build_transforms(train=True):
    """
    Build image transformation pipelines for training and evaluation.
    
    CRITICAL: Uses CONSISTENT [-1, 1] normalization to match generator output.
    """
    # CONSISTENT normalization: [0, 1] → [-1, 1]
    # Formula: x' = (x - 0.5) / 0.5 = 2x - 1
    norm = transforms.Normalize(mean=0.5, std=0.5)
    
    if train:
        aug = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),  # [0, 1]
            norm,  # → [-1, 1]
        ])
    else:
        aug = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),  # [0, 1]
            norm,  # → [-1, 1]
        ])
    
    return aug

def build_loaders(dataroot, batchsize=32, numworkers=4):
    """
    Build PyTorch DataLoaders for training, validation, and testing.
    
    CRITICAL FIX: Uses WeightedRandomSampler for CLASS BALANCING.
    This ensures all classes are sampled equally, preventing mode collapse.
    """
    # Training dataset
    trainds = datasets.ImageFolder(
        os.path.join(dataroot, 'train'),
        transform=build_transforms(True)
    )
    
    # ===== CRITICAL: Add class weights for balanced sampling =====
    class_counts = {}
    for _, label in trainds.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"\nClass distribution in training data:")
    for cls_id, count in sorted(class_counts.items()):
        print(f"  Class {cls_id}: {count} images")
    
    # Inverse weights: rare classes get higher weights
    # This ensures each class is sampled equally often
    class_weights = [1.0 / class_counts[label] for _, label in trainds.samples]
    sampler = WeightedRandomSampler(
        weights=class_weights,
        num_samples=len(trainds),
        replacement=True
    )
    
    print(f"WeightedRandomSampler: balances classes during training")
    print(f"Each epoch guarantees equal sampling of all {len(class_counts)} classes\n")
    
    # Use sampler instead of shuffle=True
    traindl = DataLoader(
        trainds,
        batch_size=batchsize,
        sampler=sampler,  # <-- KEY CHANGE: balanced sampling
        num_workers=numworkers,
        pin_memory=True
    )
    
    # Validation dataset (no balancing needed for validation)
    valds = datasets.ImageFolder(
        os.path.join(dataroot, 'valid'),
        transform=build_transforms(False)
    )
    valdl = DataLoader(valds, batch_size=batchsize, shuffle=False,
                       num_workers=numworkers, pin_memory=True)
    
    # Test dataset (no balancing needed for test)
    testds = datasets.ImageFolder(
        os.path.join(dataroot, 'test'),
        transform=build_transforms(False)
    )
    testdl = DataLoader(testds, batch_size=batchsize, shuffle=False,
                        num_workers=numworkers, pin_memory=True)
    
    return traindl, valdl, testdl, trainds.classes
