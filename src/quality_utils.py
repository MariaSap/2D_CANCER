import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import glob
import shutil
import csv
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image


# Standard size for all medical images to ensure consistency
STANDARD_IMG_SIZE = (256, 256)


def load_images_from_folder(folder_path, resize_to=STANDARD_IMG_SIZE):
    """
    Load all images from a folder as numpy arrays (grayscale).
    
    Resizes all images to a standard size to handle variable image dimensions.
    
    Args:
        folder_path: Path to folder containing images
        resize_to: Tuple (height, width) to resize images to. Default (256, 256)
    
    Returns:
        numpy array of images with shape (N, H, W) or None if no images found
    """
    images = []
    image_paths = glob.glob(os.path.join(folder_path, '*.png')) + \
                  glob.glob(os.path.join(folder_path, '*.jpg')) + \
                  glob.glob(os.path.join(folder_path, '*.jpeg'))
    
    for img_path in sorted(image_paths):
        try:
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            
            # Resize to standard size to handle variable dimensions
            img = img.resize(resize_to, Image.Resampling.LANCZOS)
            
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            images.append(img_array)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    if not images:
        return None
    
    # Stack into single array - all images now have same shape
    return np.array(images, dtype=np.float32)

def calculate_ssim_score(real_images, synthetic_images):
    """
    Calculate average SSIM between real and synthetic images.
    
    SSIM (Structural Similarity Index) measures perceived image quality
    and is more appropriate for medical images than FID.
    
    Args:
        real_images: numpy array of real images (N, H, W)
        synthetic_images: numpy array of synthetic images (M, H, W)
    
    Returns:
        float: Average SSIM score (higher is better, range 0-1)
    """
    # Match the number of images for comparison
    min_count = min(len(real_images), len(synthetic_images))
    
    ssim_scores = []
    for i in range(min_count):
        real_img = real_images[i]
        synth_img = synthetic_images[i]
        
        # Ensure same shape
        if real_img.shape != synth_img.shape:
            synth_img = np.array(Image.fromarray((synth_img * 255).astype(np.uint8)).resize(
                (real_img.shape[1], real_img.shape[0]))) / 255.0
        
        # Calculate SSIM (data_range=1.0 for normalized [0,1] images)
        score = ssim(real_img, synth_img, data_range=1.0)
        ssim_scores.append(score)
    
    return np.mean(ssim_scores) if ssim_scores else 0.0


def calculate_psnr_score(real_images, synthetic_images):
    """
    Calculate average PSNR between real and synthetic images.
    
    PSNR (Peak Signal-to-Noise Ratio) measures pixel-level fidelity.
    Higher values indicate better quality.
    
    Args:
        real_images: numpy array of real images (N, H, W)
        synthetic_images: numpy array of synthetic images (M, H, W)
    
    Returns:
        float: Average PSNR score in dB (higher is better, typically 20-40)
    """
    min_count = min(len(real_images), len(synthetic_images))
    
    psnr_scores = []
    for i in range(min_count):
        real_img = real_images[i]
        synth_img = synthetic_images[i]
        
        # Ensure same shape
        if real_img.shape != synth_img.shape:
            synth_img = np.array(Image.fromarray((synth_img * 255).astype(np.uint8)).resize(
                (real_img.shape[1], real_img.shape[0]))) / 255.0
        
        # Calculate PSNR (data_range=1.0 for normalized [0,1] images)
        score = psnr(real_img, synth_img, data_range=1.0)
        psnr_scores.append(score)
    
    return np.mean(psnr_scores) if psnr_scores else 0.0


def evaluate_synthetic_quality(real_dir, synthetic_dir, device='cuda', batch_size=32):
    """
    Evaluate the quality of synthetic images compared to real images using SSIM and PSNR.
    
    These metrics are more appropriate for medical images than FID.
    
    Args:
        real_dir: Directory containing real images organized by class
        synthetic_dir: Directory containing synthetic images organized by class
        device: Computing device (not used for SSIM/PSNR but kept for API compatibility)
        batch_size: Batch size (not used for SSIM/PSNR but kept for API compatibility)
    
    Returns:
        dict: Quality metrics including SSIM and PSNR scores per class and overall
    """
    print("Loading images for quality evaluation...")
    
    quality_scores = {}
    classes = sorted(os.listdir(real_dir))
    
    for cls in classes:
        real_cls_path = os.path.join(real_dir, cls)
        synth_cls_path = os.path.join(synthetic_dir, cls)
        
        if not os.path.exists(synth_cls_path):
            print(f"Warning: No synthetic images found for class {cls}")
            continue
        
        print(f"Evaluating quality for class {cls}...")
        
        # Load images
        real_images = load_images_from_folder(real_cls_path)
        synthetic_images = load_images_from_folder(synth_cls_path)
        
        if real_images is None or len(real_images) == 0:
            print(f"No real images found for class {cls}")
            continue
        
        if synthetic_images is None or len(synthetic_images) == 0:
            print(f"No synthetic images found for class {cls}")
            continue
        
        # Calculate metrics
        ssim_score = calculate_ssim_score(real_images, synthetic_images)
        psnr_score = calculate_psnr_score(real_images, synthetic_images)
        
        quality_scores[cls] = {
            'ssim': ssim_score,
            'psnr': psnr_score,
            'real_count': len(real_images),
            'synth_count': len(synthetic_images)
        }
        
        print(f"Class {cls}: SSIM={ssim_score:.4f}, PSNR={psnr_score:.2f}dB "
              f"(Real: {len(real_images)}, Synthetic: {len(synthetic_images)})")
    
    return quality_scores


def filter_synthetic_by_quality(synth_dir, ssim_threshold=0.5, psnr_threshold=20.0, real_dir=None):
    """
    Filter synthetic images by quality, removing low-quality samples based on SSIM/PSNR.
    
    Args:
        synth_dir: Directory containing synthetic images
        ssim_threshold: Minimum acceptable SSIM score (0-1, higher is better)
        psnr_threshold: Minimum acceptable PSNR score in dB (higher is better)
        real_dir: Directory containing real images for comparison (optional)
    
    Returns:
        None (modifies synthetic_dir in place)
    """
    print(f"Filtering synthetic images with SSIM threshold {ssim_threshold} and PSNR threshold {psnr_threshold}dB")
    
    if real_dir is None:
        print("Warning: No real data provided for quality comparison.")
        print("Using basic quality filtering based on image statistics...")
        
        removed_count = 0
        total_count = 0
        
        for cls in os.listdir(synth_dir):
            cls_path = os.path.join(synth_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            
            images = glob.glob(os.path.join(cls_path, '*.png'))
            total_count += len(images)
            
            for img_path in images:
                try:
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img)
                    
                    # Remove very dark or very bright images (likely artifacts)
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
        # Use metric-based filtering
        quality_scores = evaluate_synthetic_quality(real_dir, synth_dir)
        removed_count = 0
        total_count = 0
        
        for cls, scores in quality_scores.items():
            ssim_score = scores['ssim']
            psnr_score = scores['psnr']
            cls_path = os.path.join(synth_dir, cls)
            
            if ssim_score < ssim_threshold or psnr_score < psnr_threshold:
                images = glob.glob(os.path.join(cls_path, '*.png'))
                for img_path in images:
                    os.remove(img_path)
                    removed_count += 1
                print(f"Removed all {len(images)} images from class {cls} "
                      f"(SSIM={ssim_score:.4f} < {ssim_threshold}, PSNR={psnr_score:.2f}dB < {psnr_threshold}dB)")
            else:
                print(f"Keeping class {cls} images (SSIM={ssim_score:.4f}, PSNR={psnr_score:.2f}dB)")
            
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
            image_files = glob.glob(os.path.join(cls_path, '*'))
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
        'stddev': np.std(list(class_counts.values()))
    }
    
    return balance_stats


def save_quality_report_csv(quality_scores, num_of_epch=None, num_of_iters=None,
                           balance_before=None, balance_after=None, output_dir='.', run_name=None):
    """
    Save quality assessment results with data balance info to CSV.
    
    Uses SSIM and PSNR instead of FID for medical image evaluation.
    
    Args:
        quality_scores: Dictionary from evaluate_synthetic_quality function
        num_of_epch: Number of epochs (optional)
        num_of_iters: Number of iterations (optional)
        balance_before: Dictionary from assess_class_balance before filtering (optional)
        balance_after: Dictionary from assess_class_balance after filtering (optional)
        output_dir: Where to save the CSV
        run_name: Optional name for the run
    
    Returns:
        str: Path to the saved CSV file
    """
    import pandas as pd
    
    timestamp = datetime.now(timezone.utc)
    date_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Create timestamp
    csv_data = []
    
    for class_name, scores in quality_scores.items():
        ssim_score = scores['ssim']
        psnr_score = scores['psnr']
        
        synth_before = balance_before.get('class_counts', {}).get(class_name, 0) if balance_before else scores['synth_count']
        synth_after = balance_after.get('class_counts', {}).get(class_name, 0) if balance_after else scores['synth_count']
        
        row = {
            'RunDate': timestamp.strftime('%Y-%m-%d'),
            'RunTime': timestamp.strftime('%H:%M:%S'),
            'RunName': run_name or f'gan_eval_{date_str}',
            'Number of Epochs': num_of_epch,
            'Number of Iterations': num_of_iters,
            'Class': class_name,
            'SSIM Score': round(ssim_score, 4),
            'PSNR Score (dB)': round(psnr_score, 2),
            'Quality Grade': 'Excellent' if ssim_score > 0.8 else 'Good' if ssim_score > 0.7 
                            else 'Fair' if ssim_score > 0.6 else 'Poor',
            'Passed Threshold': 'Yes' if ssim_score > 0.6 else 'No',
            'Real Images': scores['real_count'],
            'Synthetic Before Filter': synth_before,
            'Synthetic After Filter': synth_after,
            'Images Removed': synth_before - synth_after,
            'Retention Rate (%)': round(synth_after / synth_before * 100, 1) if synth_before > 0 else 0,
            'Synth to Real Ratio': round(synth_after / scores['real_count'], 2) if scores['real_count'] > 0 else 0
        }
        csv_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(csv_data)
    
    # Add summary row
    if csv_data:
        ssim_scores = [scores['ssim'] for scores in quality_scores.values()]
        psnr_scores = [scores['psnr'] for scores in quality_scores.values()]
        total_real = sum(scores['real_count'] for scores in quality_scores.values())
        total_synth_before = sum(row['Synthetic Before Filter'] for row in csv_data)
        total_synth_after = sum(row['Synthetic After Filter'] for row in csv_data)
        
        summary_row = {
            'RunDate': timestamp.strftime('%Y-%m-%d'),
            'RunTime': timestamp.strftime('%H:%M:%S'),
            'RunName': run_name or f'gan_eval_{date_str}',
            'Number of Epochs': num_of_epch,
            'Number of Iterations': num_of_iters,
            'Class': 'OVERALL',
            'SSIM Score': round(np.mean(ssim_scores), 4),
            'PSNR Score (dB)': round(np.mean(psnr_scores), 2),
            'Quality Grade': 'Good' if np.mean(ssim_scores) > 0.7 else 'Fair' if np.mean(ssim_scores) > 0.6 else 'Poor',
            'Passed Threshold': sum(1 for s in ssim_scores if s > 0.6) / len(ssim_scores),
            'Real Images': total_real,
            'Synthetic Before Filter': total_synth_before,
            'Synthetic After Filter': total_synth_after,
            'Images Removed': total_synth_before - total_synth_after,
            'Retention Rate (%)': round(total_synth_after / total_synth_before * 100, 1) if total_synth_before > 0 else 0,
            'Synth to Real Ratio': round(total_synth_after / total_real, 2) if total_real > 0 else 0
        }
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Add balance metrics row if available
    if balance_after and 'class_counts' in balance_after:
        counts = [c for c in balance_after['class_counts'].values() if c > 0]
        if len(counts) > 1:
            balance_metrics = {
                'RunDate': timestamp.strftime('%Y-%m-%d'),
                'RunTime': timestamp.strftime('%H:%M:%S'),
                'RunName': run_name or f'gan_eval_{date_str}',
                'Class': 'BALANCE_METRICS',
                'SSIM Score': '',
                'PSNR Score (dB)': '',
                'Quality Grade': '',
                'Passed Threshold': '',
                'Real Images': '',
                'Synthetic Before Filter': f"Min: {min(counts)}, Max: {max(counts)}",
                'Synthetic After Filter': '',
                'Images Removed': f"Imbalance Ratio: {round(max(counts) / min(counts), 2)}",
                'Retention Rate (%)': f"Classes Remaining: {len(counts)}"
            }
            df = pd.concat([df, pd.DataFrame([balance_metrics])], ignore_index=True)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    filename = f'gan_quality_report_{date_str}.csv'
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"Quality report saved to {filepath}")
    return filepath


# Example usage
if __name__ == '__main__':
    DATA_ROOT = Path(r"C:data")  
    SYNTH_ROOT = Path(r"C:data\synth_2nd\train") 
    MERGED_ROOT = Path(r"C:data\synth_merged_2nd\train")  

    DATA_ROOT = r'C:\data'  # Root directory with train/valid/test folders
    SYNTH_ROOT = r'C:\data'  # Where synthetic images will be saved
    MERGED_ROOT = r'C:\data\train'  
    
    print("=" * 60)
    print("SSIM/PSNR-based Medical Image Quality Evaluation")
    print("=" * 60)
    
    # Get balance before filtering
    balance_before = assess_class_balance(os.path.join(DATA_ROOT, 'train'))
    print("\nBalance before filtering:")
    for cls, count in balance_before.get('class_counts', {}).items():
        print(f"  {cls}: {count} images")
    
    # Evaluate synthetic quality using SSIM and PSNR
    quality_scores = evaluate_synthetic_quality(
        real_dir=os.path.join(DATA_ROOT, 'train'),
        synth_dir=SYNTH_ROOT
    )
    
    # Filter low-quality synthetic images
    filter_synthetic_by_quality(
        SYNTH_ROOT,
        ssim_threshold=0.6,
        psnr_threshold=20.0,
        real_dir=os.path.join(DATA_ROOT, 'train')
    )
    
    # Get balance after filtering
    balance_after = assess_class_balance(SYNTH_ROOT)
    print("\nBalance after filtering:")
    for cls, count in balance_after.get('class_counts', {}).items():
        print(f"  {cls}: {count} images")
    
    # Evaluate again after filtering
    quality_scores = evaluate_synthetic_quality(
        real_dir=os.path.join(DATA_ROOT, 'train'),
        synth_dir=SYNTH_ROOT
    )
    
    # Save comprehensive report
    csv_file = save_quality_report_csv(
        quality_scores,
        num_of_epch=4,
        num_of_iters=3,
        balance_before=balance_before,
        balance_after=balance_after,
        output_dir='quality_reports',
        run_name='medical_gan_v1'
    )
    
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)