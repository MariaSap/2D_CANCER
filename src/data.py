import os, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMG_SIZE = 256

def build_transforms(train=True):
    norm = transforms.Normalize(mean=[0.5], std=[0.5])
    if train:
        aug = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([IMG_SIZE, IMG_SIZE]),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9,1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            norm,
        ])

    else:
        aug = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            norm,
        ])

    return aug


def build_loaders(data_root, batch_size=32, num_workers=4):
    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=build_transforms(True))
    val_ds   = datasets.ImageFolder(os.path.join(data_root, "valid"),   transform=build_transforms(False))
    test_ds  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=build_transforms(False))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, test_dl, train_ds.classes