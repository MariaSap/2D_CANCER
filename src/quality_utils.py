# Quality Assessment Utilities for Medical Image GANs
# This module provides functions to evaluate and filter synthetic medical images

import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
import glob
import shutil

def load_inception_model(device='cpu'):
    """
    Load pre-trained Inception-v3 model for feature extraction.
    Used for calculating FID (Fréchet Inception Distance) scores.
    """
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.eval()
    model.to(device)
    return model

def extract_inception_features(images, model, device='cpu'):
    """
    Extract features from images using Inception-v3 model.
    
    Args:
        images: Batch of images [batch_size, channels, height, width]
        model: Pre-trained Inception model
        device: Computing device
    
    Returns:
        numpy array of features [batch_size, 2048]
    """
    model.eval()
    with torch.no_grad():
        # Resize images to 299x299 for Inception
        if images.shape[-1] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Convert grayscale to RGB if needed
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Get features from the last pooling layer
        features = model(images.to(device))
        
        # Handle different Inception outputs
        if isinstance(features, tuple):
            features = features[0]
        
        return features.cpu().numpy()

def calculate_fid_score(real_features, fake_features):
    """
    Calculate Fréchet Inception Distance between real and fake image features.
    Lower FID scores indicate better synthetic image quality.
    
    Args:
        real_features: Features from real images [N, feature_dim]
        fake_features: Features from synthetic images [M, feature_dim]
    
    Returns:
        float: FID score
    """
    # Calculate means and covariances
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fid

def evaluate_synthetic_quality(real_dir, synthetic_dir, device='cpu', batch_size=32):
    """
    Evaluate the quality of synthetic images compared to real images using FID.
    
    Args:
        real_dir: Directory containing real images
        synthetic_dir: Directory containing synthetic images
        device: Computing device
        batch_size: Batch size for processing
    
    Returns:
        dict: Quality metrics including FID score per class and overall
    """
    print("Loading Inception model for quality evaluation...")
    inception_model = load_inception_model(device)
    
    # Image preprocessing for Inception
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    quality_scores = {}
    
    # Get list of classes
    classes = sorted(os.listdir(real_dir))
    
    for cls in classes:
        real_cls_path = os.path.join(real_dir, cls)
        synth_cls_path = os.path.join(synthetic_dir, cls)
        
        if not os.path.exists(synth_cls_path):
            print(f"Warning: No synthetic images found for class {cls}")
            continue
        
        print(f"Evaluating quality for class {cls}...")
        
        # Load real images
        real_dataset = ImageFolder(real_dir, transform=transform)
        real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
        
        # Load synthetic images
        synth_dataset = ImageFolder(synthetic_dir, transform=transform)
        synth_loader = DataLoader(synth_dataset, batch_size=batch_size, shuffle=False)
        
        # Extract features from real images
        real_features = []
        for images, labels in real_loader:
            # Only process images from current class
            class_mask = (labels == int(cls))
            if class_mask.any():
                class_images = images[class_mask]
                features = extract_inception_features(class_images, inception_model, device)
                real_features.append(features)
        
        if real_features:
            real_features = np.concatenate(real_features, axis=0)
        else:
            print(f"No real images found for class {cls}")
            continue
        
        # Extract features from synthetic images
        synth_features = []
        for images, labels in synth_loader:
            # Only process images from current class
            class_mask = (labels == int(cls))
            if class_mask.any():
                class_images = images[class_mask]
                features = extract_inception_features(class_images, inception_model, device)
                synth_features.append(features)
        
        if synth_features:
            synth_features = np.concatenate(synth_features, axis=0)
            
            # Calculate FID score
            fid_score = calculate_fid_score(real_features, synth_features)
            quality_scores[cls] = {
                'fid': fid_score,
                'real_count': len(real_features),
                'synth_count': len(synth_features)
            }
            
            print(f"Class {cls}: FID = {fid_score:.3f} "
                  f"(Real: {len(real_features)}, Synthetic: {len(synth_features)})")
        else:
            print(f"No synthetic images found for class {cls}")
    
    return quality_scores

def filter_synthetic_by_quality(synth_dir, fid_threshold=50.0, real_dir=None):
    """
    Filter synthetic images by quality, removing low-quality samples.
    
    Args:
        synth_dir: Directory containing synthetic images
        fid_threshold: Maximum allowed FID score (lower is better)
        real_dir: Directory containing real images for comparison (optional)
    """
    print(f"Filtering synthetic images with FID threshold: {fid_threshold}")
    
    if real_dir is None:
        # Simple filtering based on individual image quality metrics
        # This is a placeholder - you can implement more sophisticated metrics
        print("Warning: No real data provided for FID calculation.")
        print("Using basic quality filtering...")
        
        removed_count = 0
        total_count = 0
        
        for cls in os.listdir(synth_dir):
            cls_path = os.path.join(synth_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            
            images = glob.glob(os.path.join(cls_path, "*.png"))
            total_count += len(images)
            
            # Simple quality check: remove very dark or very bright images
            # In practice, you would implement more sophisticated quality metrics
            for img_path in images:
                try:
                    # Load and check basic statistics
                    from PIL import Image
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img_array = np.array(img)
                    
                    # Remove images that are too dark or too bright (likely artifacts)
                    mean_brightness = np.mean(img_array)
                    if mean_brightness < 10 or mean_brightness > 245:
                        os.remove(img_path)
                        removed_count += 1
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    os.remove(img_path)
                    removed_count += 1
        
        print(f"Removed {removed_count}/{total_count} low-quality synthetic images")
        
    else:
        # Use FID-based filtering with real images as reference
        quality_scores = evaluate_synthetic_quality(real_dir, synth_dir)
        
        removed_count = 0
        total_count = 0
        
        for cls, scores in quality_scores.items():
            fid_score = scores['fid']
            cls_path = os.path.join(synth_dir, cls)
            
            if fid_score > fid_threshold:
                # Remove all synthetic images from this class (poor quality)
                images = glob.glob(os.path.join(cls_path, "*.png"))
                for img_path in images:
                    os.remove(img_path)
                    removed_count += 1
                
                print(f"Removed all {len(images)} images from class {cls} "
                      f"(FID: {fid_score:.3f} > {fid_threshold})")
            else:
                print(f"Keeping class {cls} images (FID: {fid_score:.3f} <= {fid_threshold})")
            
            total_count += scores['synth_count']
        
        print(f"Quality filtering complete: removed {removed_count}/{total_count} images")

def assess_class_balance(data_dir):
    """
    Assess class balance in a dataset directory.
    
    Args:
        data_dir: Directory containing class subdirectories
        
    Returns:
        dict: Class counts and balance statistics
    """
    class_counts = {}
    
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            image_files = glob.glob(os.path.join(cls_path, "*"))
            class_counts[cls] = len(image_files)
    
    if not class_counts:
        return {}
    
    total = sum(class_counts.values())
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    
    balance_stats = {
        'class_counts': class_counts,
        'total_images': total,
        'min_count': min_count,
        'max_count': max_count,
        'imbalance_ratio': max_count / min_count if min_count > 0 else float('inf'),
        'std_dev': np.std(list(class_counts.values()))
    }
    
    return balance_stats