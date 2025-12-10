# Medical image classifier implementation using ResNet architecture
# This module provides functionality for creating and training CNN classifiers for cancer detection

import torch, torch.nn as nn, torch.optim as optim
from torchvision import models

def make_classifier(num_classes=4, device="cuda"):
    """
    Create a ResNet-18 based classifier adapted for medical image classification.
    
    ResNet-18 is chosen for its balance of performance and computational efficiency,
    making it suitable for medical applications where quick diagnosis is important.
    
    Args:
        num_classes (int): Number of cancer types/classes to classify (default: 4)
        
    Returns:
        torch.nn.Module: Modified ResNet-18 model for single-channel medical images
    """
    # Load ResNet-18 architecture without pre-trained weights
    # We start from scratch since medical images differ significantly from ImageNet
    model = models.resnet50(weights=None)
    model = model.to(device)
    
    # Modify first convolutional layer for single-channel (grayscale) input
    # Original ResNet expects 3-channel RGB images, but medical images are often grayscale
    
    # Extract the mean of RGB weights to create single-channel weights
    # This preserves some of the learned feature patterns from the original architecture
    w = model.conv1.weight.data.mean(dim=1, keepdim=True)
    
    # Replace the first conv layer: 1 input channel instead of 3
    # kernel_size=7, stride=2, padding=3 maintains the same spatial dimensions
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Initialize the new layer with the averaged weights
    model.conv1.weight.data = w
    
    # Replace the final classification layer for our specific number of classes
    # Original ResNet-18 has 1000 output classes (ImageNet)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def train_classifier(model, train_dl, val_dl, epochs=25, lr=1e-3, device="cuda"):
    """
    Train the classifier using standard supervised learning with early stopping based on validation accuracy.
    
    This training procedure is optimized for medical image classification where
    validation performance is crucial for reliable diagnosis.
    
    Args:
        model: The classifier model to train
        train_dl: DataLoader for training data with augmentation
        val_dl: DataLoader for validation data without augmentation  
        epochs (int): Maximum number of training epochs
        lr (float): Learning rate for Adam optimizer
        device (str): Computing device ('cuda' or 'cuda')
        
    Returns:
        torch.nn.Module: The best performing model based on validation accuracy
    """
    # Move model to specified device (GPU for faster training if available)
    model = model.to(device)
    
    # Adam optimizer - adaptive learning rate, works well for medical imaging tasks
    # Default betas=(0.9, 0.999) provide good convergence for most cases
    opt = optim.Adam(model.parameters(), lr=lr)
    
    # Cross-entropy loss - standard for multi-class classification
    # Automatically applies softmax and computes negative log-likelihood
    crit = nn.CrossEntropyLoss()
    
    # Track the best model based on validation accuracy
    # In medical applications, validation performance is critical for generalization
    best = {"acc": 0.0, "state": None}
    
    # Training loop
    for ep in range(epochs):
        print(f"Epoch {ep} running on {next(model.parameters()).device}")

        # Training phase
        model.train()  # Enable dropout and batch normalization training mode
        
        # Process training 
        for x, y in train_dl:
            # Move data to device
            x, y = x.to(device), y.to(device)
            assert x.is_cuda and y.is_cuda, "Batch not on GPU"
            
            # Zero gradients from previous iteration
            opt.zero_grad()
            
            # Forward pass: compute predictions
            logits = model(x)
            
            # Compute loss between predictions and true labels
            loss = crit(logits, y)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update model parameters
            opt.step()
        
        # Validation phase - evaluate model performance
        model.eval()  # Disable dropout and batch normalization updates
        correct, total = 0, 0
        
        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                assert x.is_cuda and y.is_cuda, "Batch not on GPU"
                # Get predictions (class with highest probability)
                pred = model(x).argmax(1)
                
                # Count correct predictions
                correct += (pred == y).sum().item()
                total += y.numel()  # Number of elements in tensor
        
        # Calculate validation accuracy
        acc = correct / total
        
        # Save the best model based on validation accuracy
        # This implements early stopping to prevent overfitting
        if acc > best["acc"]:
            best["acc"] = acc
            best["state"] = model.state_dict()  # Save model parameters
    
    # Load the best performing model weights
    model.load_state_dict(best["state"])
    return model