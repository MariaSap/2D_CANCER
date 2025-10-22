# Differential Augmentation implementation (brightness, contrast, translation, cutout) for GAN training stabilization
# This module provides data augmentation techniques specifically designed for GAN stabilized training
# to prevent discriminator overfitting and improve training stability

import torch
import torch.nn.functional as F

def rand_brightness(x, strength=0.05):
    """
    Apply random brightness adjustment to images.
    
    Brightness augmentation helps prevent the discriminator from overfitting to
    specific lighting conditions in medical images, improving GAN robustness.
    
    Args:
        x: Input images tensor [batch, channels, height, width]
        strength: Maximum brightness change as fraction of intensity range (default: 0.05)
        
    Returns:
        torch.Tensor: Images with random brightness adjustment
    """
    # Generate random brightness adjustment for each image in batch
    # Range: [-strength, +strength] applied uniformly across all pixels
    b = (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 2 * strength
    
    # Add brightness offset to all pixels
    # Broadcasting ensures same brightness change applied to entire image
    return x + b

def rand_contrast(x, strength=0.05):
    """
    Apply random contrast adjustment to images.
    
    Contrast augmentation simulates different imaging conditions and prevents
    the discriminator from overfitting to specific contrast levels.
    
    Args:
        x: Input images tensor [batch, channels, height, width]
        strength: Maximum contrast change as fraction (default: 0.05)
        
    Returns:
        torch.Tensor: Images with random contrast adjustment
    """
    # Calculate mean intensity for each image (used as contrast pivot point)
    # keepdim=True preserves dimensions for broadcasting
    mean = x.mean(dim=[2,3], keepdim=True)
    
    # Generate random contrast multiplier for each image in batch
    # Range: [1-strength, 1+strength] where 1.0 = no change
    c = 1.0 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 2 * strength
    
    # Apply contrast adjustment: new_intensity = mean + contrast * (old_intensity - mean)
    # This preserves the mean while scaling deviations from the mean
    return (x - mean) * c + mean

def rand_translate(x, ratio=0.05):
    """
    Apply random translation (shifting) to images.
    
    Translation augmentation simulates different positioning of subjects in medical
    images, helping the GAN generate more diverse and realistic variations.
    
    Args:
        x: Input images tensor [batch, channels, height, width]
        ratio: Maximum translation as fraction of image size (default: 0.05 = 5%)
        
    Returns:
        torch.Tensor: Images with random translation applied
    """
    b, c, h, w = x.shape
    
    # Generate random horizontal shifts for each image in batch
    # Range: [-w*ratio, +w*ratio] pixels
    shift_x = (torch.randint(int(-w*ratio), int(w*ratio)+1, (b,), device=x.device)).float()
    
    # Generate random vertical shifts for each image in batch  
    # Range: [-h*ratio, +h*ratio] pixels
    shift_y = (torch.randint(int(-h*ratio), int(h*ratio)+1, (b,), device=x.device)).float()
    
    # Create coordinate grids for sampling
    # meshgrid creates x,y coordinate matrices for the entire image
    grid_x, grid_y = torch.meshgrid(
        torch.arange(w, device=x.device), 
        torch.arange(h, device=x.device), 
        indexing='xy'
    )
    
    # Apply translation by shifting the sampling coordinates
    # Normalize to [-1, 1] range required by grid_sample
    grid_x = (grid_x.unsqueeze(0) + shift_x.view(-1,1,1)) / (w-1) * 2 - 1
    grid_y = (grid_y.unsqueeze(0) + shift_y.view(-1,1,1)) / (h-1) * 2 - 1
    
    # Stack x,y coordinates into sampling grid
    grid = torch.stack((grid_x, grid_y), dim=-1)
    
    # Sample from original image using transformed coordinates
    # bilinear interpolation for smooth results
    # reflection padding handles out-of-bounds coordinates
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)

def rand_cutout(x, ratio=0.2):
    """
    Apply random cutout (masking) to images.
    
    Cutout augmentation forces the discriminator to not rely on specific image
    regions, improving robustness and preventing overfitting to particular
    anatomical structures in medical images.
    
    Args:
        x: Input images tensor [batch, channels, height, width]
        ratio: Maximum cutout size as fraction of image dimensions (default: 0.2 = 20%)
        
    Returns:
        torch.Tensor: Images with random rectangular regions masked to zero
    """
    b, c, h, w = x.shape
    
    # Generate random cutout dimensions for this batch
    # Size varies from 0 to ratio*image_size in each dimension
    cut_h = int(h * ratio * torch.rand(1).item())
    cut_w = int(w * ratio * torch.rand(1).item())
    
    # Generate random cutout positions for each image in batch
    # Ensure cutout doesn't go outside image boundaries
    y = torch.randint(0, h - cut_h + 1, (b,), device=x.device)  # Top-left y coordinate
    x0 = torch.randint(0, w - cut_w + 1, (b,), device=x.device)  # Top-left x coordinate
    
    # Create mask: 1 for keep, 0 for remove
    mask = torch.ones_like(x)
    
    # Apply cutout for each image individually (different positions per image)
    for i in range(b):
        mask[i, :, y[i]:y[i]+cut_h, x0[i]:x0[i]+cut_w] = 0
    
    # Apply mask to zero out the cutout regions
    return x * mask

def diffaugment(x):
    """
    Apply differential augmentation pipeline to images.
    
    Combines multiple augmentation techniques in a specific order optimized for
    GAN training stability. This prevents discriminator overfitting while
    maintaining the visual quality needed for medical image generation.
    
    The augmentation pipeline is applied identically to both real and fake images
    during discriminator training, forcing it to focus on meaningful differences
    rather than augmentation artifacts.
    
    Args:
        x: Input images tensor [batch, channels, height, width]
        
    Returns:
        torch.Tensor: Augmented images with same shape as input
    """
    # Apply augmentations in order of increasing invasiveness
    
    # 1. Geometric transformation: translation (least destructive)
    x = rand_translate(x, 0.05)  # Up to 5% translation
    
    # 2. Spatial masking: cutout (removes information but preserves rest)
    x = rand_cutout(x, 0.15)     # Up to 15% area cutout
    
    # 3. Intensity adjustments: brightness and contrast (most global changes)
    x = rand_brightness(x, 0.05)  # Up to 5% brightness change
    x = rand_contrast(x, 0.05)    # Up to 5% contrast change
    
    return x