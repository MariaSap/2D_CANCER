# GAN sampling utilities for generating synthetic medical images
# This module handles the generation of synthetic cancer images from trained GAN models
"""
 Generates synthetic medical images from trained GANs, organizing them by class for 
 balanced data augmentation. Essential for creating large-scale synthetic datasets to 
 supplement limited medical data.
"""
import os, torch
from torchvision.utils import save_image
from gan import Generator

def sample_to_folder(g_ema, out_root, per_class=2000, z_dim=128, num_classes=4, device="cpu"):
    """
    Generate synthetic images from a trained GAN and save them organized by class.
    
    This function is crucial for data augmentation in medical imaging where obtaining
    large datasets is challenging due to privacy concerns and rarity of certain conditions.
    
    Args:
        g_ema: Exponential Moving Average version of the trained Generator model
               (EMA provides more stable and higher quality generations)
        out_root (str): Root directory where synthetic images will be saved
        per_class (int): Number of synthetic images to generate per class (default: 2000)
        z_dim (int): Dimension of the noise vector input (latent space dimension)
        num_classes (int): Number of different cancer types/classes to generate
        device (str): Computing device ('cuda' or 'cpu')
    """
    # Create the output directory if it doesn't exist    
    os.makedirs(out_root, exist_ok=True)

    # Generate images for each class separately to ensure balanced synthetic dataset
    for c in range(num_classes):
        # Create subdirectory for each class (e.g., benign, malignant, etc.)
        cls_dir = os.path.join(out_root, str(c))
        os.makedirs(cls_dir, exist_ok=True)
        
        # Counter for tracking number of generated images for this class
        n = 0 

       # Generate images in batches until we reach the target per_class count
        while n < per_class:
            # Calculate batch size - use 64 or remaining images needed, whichever is smaller
            # Batch processing is memory-efficient and speeds up generation
            bs = min(64, per_class - n)
            
            # Sample random noise vectors from standard normal distribution
            # This is the input to the generator - each z vector will produce a unique image
            z = torch.randn(bs, z_dim, device=device)
            
            # Create class labels - all images in this batch belong to class 'c'
            # torch.full creates a tensor filled with the value 'c'
            y = torch.full((bs,), c, device=device, dtype=torch.long)
            
            # Generate synthetic images
            with torch.no_grad():  # Disable gradient computation for inference (saves memory)
                # Pass noise and class labels through the generator
                x = g_ema(z, y)
                
                # Convert from tanh output range [-1,1] to image range [0,1]
                # tanh is commonly used in GANs to ensure bounded output
                x = (x.clamp(-1,1)+1)/2.0
            
            # Save each generated image as a separate file
            for i in range(bs):
                # Save with zero-padded filename for consistent ordering (e.g., 000001.png)
                save_image(x[i], os.path.join(cls_dir, f"{n+i:06d}.png"))
            
            # Update counter for next batch
            n += bs







# # Enhanced GAN sampling utilities with data analysis for medical image augmentation
# # This module handles generation of synthetic cancer images with intelligent minority class detection

# import os, torch, glob
# from collections import Counter
# from torchvision.utils import save_image
# from gan import Generator

# def count_files_per_class(data_root):
#     """
#     Count the number of files in each class directory to identify minority classes.
    
#     This function is crucial for medical image datasets where class imbalance is common.
#     Rare cancer types often have significantly fewer samples than common ones,
#     requiring targeted synthetic data augmentation.
    
#     Args:
#         data_root (str): Root directory containing class subdirectories
#                         Expected structure: data_root/class_0/, data_root/class_1/, etc.
        
#     Returns:
#         dict: Dictionary mapping class names to file counts
#         list: Sorted list of (class_name, count) tuples, ascending by count
#     """
#     print("Analyzing class distribution in dataset...")
    
#     # Dictionary to store counts for each class
#     class_counts = {}
    
#     # Get all subdirectories (classes) in the data root
#     if not os.path.exists(data_root):
#         print(f"Warning: Data root '{data_root}' does not exist!")
#         return {}, []
    
#     # List all subdirectories (class folders)
#     class_dirs = [d for d in os.listdir(data_root) 
#                   if os.path.isdir(os.path.join(data_root, d))]
    
#     if not class_dirs:
#         print(f"Warning: No class directories found in '{data_root}'!")
#         return {}, []
    
#     print(f"Found {len(class_dirs)} classes: {sorted(class_dirs)}\n")
    
#     # Count files in each class directory
#     for class_name in sorted(class_dirs):
#         class_path = os.path.join(data_root, class_name)
        
#         # Count all image files (common medical image formats)
#         image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.dcm']
#         file_count = 0
        
#         for ext in image_extensions:
#             file_count += len(glob.glob(os.path.join(class_path, ext)))
#             file_count += len(glob.glob(os.path.join(class_path, ext.upper())))
        
#         class_counts[class_name] = file_count
#         print(f"Class {class_name}: {file_count} images")
    
#     # Sort classes by count (ascending) to identify minorities first
#     sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    
#     # Analysis summary
#     total_images = sum(class_counts.values())
#     min_count = min(class_counts.values()) if class_counts else 0
#     max_count = max(class_counts.values()) if class_counts else 0
#     avg_count = total_images / len(class_counts) if class_counts else 0
    
#     print(f"\n=== CLASS DISTRIBUTION ANALYSIS ===")
#     print(f"Total images: {total_images}")
#     print(f"Average per class: {avg_count:.1f}")
#     print(f"Min count: {min_count} (class {sorted_classes[0][0] if sorted_classes else 'N/A'})")
#     print(f"Max count: {max_count} (class {sorted_classes[-1][0] if sorted_classes else 'N/A'})")
#     print(f"Imbalance ratio: {max_count/min_count:.2f}:1" if min_count > 0 else "N/A")
    
#     # Identify minority classes (below average)
#     minority_classes = [cls for cls, count in sorted_classes if count < avg_count]
#     if minority_classes:
#         print(f"Minority classes (below average): {minority_classes}")
    
#     print("=" * 45)
    
#     return class_counts, sorted_classes

# def calculate_augmentation_needs(class_counts, target_count=None, balance_strategy='max'):
#     """
#     Calculate how many synthetic images to generate for each class based on imbalance.
    
#     Args:
#         class_counts (dict): Dictionary mapping class names to current file counts
#         target_count (int, optional): Target number of images per class
#         balance_strategy (str): Strategy for balancing ('max', 'mean', 'median', or custom target)
        
#     Returns:
#         dict: Dictionary mapping class names to number of synthetic images needed
#     """
#     if not class_counts:
#         return {}
    
#     counts = list(class_counts.values())
    
#     # Determine target count based on strategy
#     if target_count is None:
#         if balance_strategy == 'max':
#             target_count = max(counts)
#         elif balance_strategy == 'mean':
#             target_count = int(sum(counts) / len(counts))
#         elif balance_strategy == 'median':
#             sorted_counts = sorted(counts)
#             n = len(sorted_counts)
#             target_count = sorted_counts[n//2] if n % 2 == 1 else int((sorted_counts[n//2-1] + sorted_counts[n//2]) / 2)
#         else:
#             target_count = max(counts)  # Default to max
    
#     print(f"\nAugmentation strategy: {balance_strategy}")
#     print(f"Target images per class: {target_count}")
    
#     # Calculate how many synthetic images needed for each class
#     augmentation_needs = {}
#     for class_name, current_count in class_counts.items():
#         needed = max(0, target_count - current_count)
#         augmentation_needs[class_name] = needed
        
#         if needed > 0:
#             print(f"Class {class_name}: need {needed} synthetic images ({current_count} -> {target_count})")
#         else:
#             print(f"Class {class_name}: no augmentation needed ({current_count} >= {target_count})")
    
#     return augmentation_needs

# def sample_to_folder(g_ema, out_root, data_root, z_dim=128, num_classes=4, 
#                             device="cuda", balance_strategy='max', target_count=None):
#     """
#     Generate synthetic images with intelligent class balancing based on existing data distribution.
    
#     This enhanced version analyzes the current dataset, identifies minority classes,
#     and generates synthetic data accordingly to create a more balanced training set.
    
#     Args:
#         g_ema: Exponential Moving Average version of the trained Generator
#         out_root (str): Root directory where synthetic images will be saved
#         data_root (str): Root directory with existing real images (for analysis)
#         z_dim (int): Dimension of the noise vector input
#         num_classes (int): Number of different cancer types/classes
#         device (str): Computing device ('cuda' or 'cpu')
#         balance_strategy (str): How to balance classes ('max', 'mean', 'median')
#         target_count (int, optional): Specific target count per class
#     """
#     print("Starting intelligent synthetic data generation...")
    
#     # Step 1: Analyze existing data distribution
#     class_counts, sorted_classes = count_files_per_class(data_root)
    
#     # if not class_counts:
#     #     print("No existing data found. Using default generation settings.")
#     #     # Fallback to original behavior
#     #     sample_to_folder(g_ema, out_root, per_class=2000, z_dim=z_dim, 
#     #                             num_classes=num_classes, device=device)
#     #     return
    
#     # Step 2: Calculate augmentation needs
#     augmentation_needs = calculate_augmentation_needs(class_counts, target_count, balance_strategy)
    
#     # Step 3: Generate synthetic images based on needs
#     os.makedirs(out_root, exist_ok=True)
    
#     total_to_generate = sum(augmentation_needs.values())
#     if total_to_generate == 0:
#         print("Dataset is already balanced. No synthetic images needed.")
#         return
    
#     print(f"\nGenerating {total_to_generate} synthetic images total...")
    
#     generated_count = 0
#     for class_idx in range(num_classes):
#         class_name = str(class_idx)
#         needed = augmentation_needs.get(class_name, 0)
        
#         if needed == 0:
#             print(f"Skipping class {class_name} - no augmentation needed")
#             continue
        
#         print(f"Generating {needed} synthetic images for class {class_name}...")
        
#         # Create class directory
#         cls_dir = os.path.join(out_root, class_name)
#         os.makedirs(cls_dir, exist_ok=True)
        
#         # Generate images in batches
#         n = 0
#         while n < needed:
#             # Calculate batch size (up to 64, or remaining needed)
#             bs = min(64, needed - n)
            
#             # Sample random noise vectors
#             z = torch.randn(bs, z_dim, device=device)
            
#             # Create class labels for this batch
#             y = torch.full((bs,), class_idx, device=device, dtype=torch.long)
            
#             # Generate synthetic images
#             with torch.no_grad():
#                 x = g_ema(z, y)
#                 # Convert from tanh range [-1,1] to image range [0,1]
#                 x = (x.clamp(-1,1)+1)/2.0
            
#             # Save each generated image
#             for i in range(bs):
#                 save_image(x[i], os.path.join(cls_dir, f"synth_{n+i:06d}.png"))
            
#             n += bs
#             generated_count += bs
            
#             # Progress update
#             if n % 256 == 0 or n >= needed:
#                 print(f"  Generated {n}/{needed} images for class {class_name}")
    
#     print(f"\n✓ Successfully generated {generated_count} synthetic images")
#     print(f"✓ Synthetic images saved to: {out_root}")
    
#     # Final summary
#     print("\n=== AUGMENTATION SUMMARY ===")
#     for class_name, original_count in class_counts.items():
#         synthetic_count = augmentation_needs.get(class_name, 0)
#         total_count = original_count + synthetic_count
#         print(f"Class {class_name}: {original_count} real + {synthetic_count} synthetic = {total_count} total")

# # def sample_to_folder(g_ema, out_root, per_class=2000, z_dim=128, num_classes=4, device="cuda"):
# #     """
# #     Original function: Generate fixed number of synthetic images per class.
    
# #     Kept for backward compatibility and cases where uniform generation is desired.
# #     """
# #     os.makedirs(out_root, exist_ok=True)
    
# #     for c in range(num_classes):
# #         cls_dir = os.path.join(out_root, str(c))
# #         os.makedirs(cls_dir, exist_ok=True)
        
# #         n = 0
# #         while n < per_class:
# #             bs = min(64, per_class - n)
# #             z = torch.randn(bs, z_dim, device=device)
# #             y = torch.full((bs,), c, device=device, dtype=torch.long)
            
# #             with torch.no_grad():
# #                 x = g_ema(z, y)
# #                 x = (x.clamp(-1,1)+1)/2.0
            
# #             for i in range(bs):
# #                 save_image(x[i], os.path.join(cls_dir, f"{n+i:06d}.png"))
            
# #             n += bs
