import os
from torchvision import datasets

DATA_DIR = './data' # Make sure this points to your data

def check_label_consistency():
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'))
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'valid'))
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'))

    print(f"Train Mapping: {train_ds.class_to_idx}")
    print(f"Valid Mapping: {val_ds.class_to_idx}")
    print(f"Test  Mapping: {test_ds.class_to_idx}")

    if train_ds.class_to_idx == test_ds.class_to_idx == val_ds.class_to_idx:
        print("\n✅ Labels are ALIGNED. The issue is likely model overfitting.")
    else:
        print("\n❌ CRITICAL WARNING: Label Mismatch Detected!")
        
check_label_consistency()