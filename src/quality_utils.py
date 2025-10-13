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
import csv
from datetime import datetime, timezone
from pathlib import Path


def load_inception_model(device='cuda'):
    """
    Load pre-trained Inception-v3 model for feature extraction.
    Used for calculating FID (Fréchet Inception Distance) scores.
    """
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.eval()
    model.to(device)
    return model

def extract_inception_features(images, model, device='cuda'):
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
        
        # # Handle different Inception outputs
        # if isinstance(features, tuple):
            # features = features[0]
        # If inception returns a tuple or InceptionOutput, take .logits or first element
        if isinstance(features, tuple):
            features = features[0]
        # Some torchvision versions return InceptionOutput with .logits
        if hasattr(features, "logits"):
            features = features.logits

        # return features.cuda().numpy()
        return features.detach().cpu().numpy()

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

def evaluate_synthetic_quality(real_dir, synthetic_dir, device='cuda', batch_size=32):
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
        
        # Map class name -> index once
        real_map = real_dataset.class_to_idx
        synth_map = synth_dataset.class_to_idx

        real_map = real_dataset.class_to_idx
        synth_map = synth_dataset.class_to_idx
        if real_map != synth_map:
            # Enforce identical indexing by reordering classes or rebuilding one dataset
            # Easiest: check same keys; use indices from real_map everywhere
            pass

        cls_idx = real_map[cls]

        # Extract features from real images
        real_features = []
        for images, labels in real_loader:
            # print(labels)
            # Only process images from current class
            class_mask = (labels == cls_idx)
            # class_mask = (labels == int(cls))
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
            class_mask = (labels == cls_idx)
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


def save_quality_report_csv(quality_scores, num_of_epch=None, num_of_iters=None, balance_before=None, balance_after=None, output_dir=".", run_name=None):
    """
    Simple function to save quality assessment results with data balance info to CSV.
    
    Args:
        quality_scores: Dictionary from your evaluate_synthetic_quality() function
        balance_before: Dictionary from assess_class_balance() before filtering (optional)
        balance_after: Dictionary from assess_class_balance() after filtering (optional)
        output_dir: Where to save the CSV
        run_name: Optional name for the run
    
    Returns:
        str: Path to the saved CSV file
    """
    import pandas as pd
    from datetime import datetime
    
    # Create timestamp
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for CSV
    csv_data = []
    
    for class_name, scores in quality_scores.items():
        # Get balance info if available
        synth_before = balance_before.get('class_counts', {}).get(class_name, 0) if balance_before else scores['synth_count']
        synth_after = balance_after.get('class_counts', {}).get(class_name, 0) if balance_after else scores['synth_count']
        
        row = {
            'Run_Date': timestamp.strftime("%Y-%m-%d"),
            'Run_Time': timestamp.strftime("%H:%M:%S"), 
            'Run_Name': run_name or f"gan_eval_{date_str}",
            'Number of Epochs': num_of_epch,
            'Number of Iterations': num_of_iters,
            'Class': class_name,
            'FID_Score': round(scores['fid'], 3),
            'Quality_Grade': 'Excellent' if scores['fid'] <= 20 else 
                 'Good' if scores['fid'] <= 40 else 
                 'Fair' if scores['fid'] <= 60 else 
                 'Poor' if scores['fid'] <= 100 else 'Unacceptable',
            'Passed_Threshold': 'Yes' if scores['fid'] <= 50.0 else 'No',
            'Real_Images': scores['real_count'],
            'Synthetic_Before_Filter': synth_before,
            'Synthetic_After_Filter': synth_after,
            'Images_Removed': synth_before - synth_after,
            'Retention_Rate': round((synth_after / synth_before * 100) if synth_before > 0 else 0, 1),
            'Synth_to_Real_Ratio': round(synth_after / scores['real_count'], 2) if scores['real_count'] > 0 else 0
        }
        csv_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(csv_data)
    
    # Add summary row
    if csv_data:
        fid_scores = [scores['fid'] for scores in quality_scores.values()]
        total_real = sum(scores['real_count'] for scores in quality_scores.values())
        total_synth_before = sum(row['Synthetic_Before_Filter'] for row in csv_data)
        total_synth_after = sum(row['Synthetic_After_Filter'] for row in csv_data)
        
        summary_row = {
            'Run_Date': timestamp.strftime("%Y-%m-%d"),
            'Run_Time': timestamp.strftime("%H:%M:%S"),
            'Run_Name': run_name or f"gan_eval_{date_str}",
            'Class': 'OVERALL',
            'FID_Score': round(sum(fid_scores) / len(fid_scores), 3),
            'Quality_Grade': 'Good' if sum(fid_scores) / len(fid_scores) <= 40 else 'Fair' if sum(fid_scores) / len(fid_scores) <= 60 else 'Poor',
            'Passed_Threshold': f"{sum(1 for fid in fid_scores if fid <= 50.0)}/{len(fid_scores)}",
            'Real_Images': total_real,
            'Synthetic_Before_Filter': total_synth_before,
            'Synthetic_After_Filter': total_synth_after,
            'Images_Removed': total_synth_before - total_synth_after,
            'Retention_Rate': round((total_synth_after / total_synth_before * 100) if total_synth_before > 0 else 0, 1),
            'Synth_to_Real_Ratio': round(total_synth_after / total_real, 2) if total_real > 0 else 0
        }
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Add balance metrics row
    if balance_after and 'class_counts' in balance_after:
        counts = [c for c in balance_after['class_counts'].values() if c > 0]
        
        if len(counts) > 1:
            balance_metrics = {
                'Run_Date': timestamp.strftime("%Y-%m-%d"),
                'Run_Time': timestamp.strftime("%H:%M:%S"),
                'Run_Name': run_name or f"gan_eval_{date_str}",
                'Class': 'BALANCE_METRICS',
                'FID_Score': '',
                'Quality_Grade': '',
                'Passed_Threshold': '',
                'Real_Images': '',
                'Synthetic_Before_Filter': '',
                'Synthetic_After_Filter': '',
                'Images_Removed': f"Min: {min(counts)}, Max: {max(counts)}",
                'Retention_Rate': f"Imbalance Ratio: {round(max(counts) / min(counts), 2)}",
                'Synth_to_Real_Ratio': f"Classes Remaining: {len(counts)}"
            }
            df = pd.concat([df, pd.DataFrame([balance_metrics])], ignore_index=True)
    
    # Save to CSV
    filename = f"gan_quality_report_{date_str}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"✅ Quality report with balance info saved to: {filepath}")
    return filepath


# import os
# import torch
# import torch.nn.functional as F
# from torchvision import models, transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# import numpy as np
# from scipy import linalg
# import pandas as pd
# import glob
# import shutil
# from datetime import datetime
# from pathlib import Path
# import json
# from PIL import Image

# # Import your existing functions (ensure they are available)
# # from your_existing_module import load_inception_model, extract_inception_features, calculate_fid_score, evaluate_synthetic_quality, assess_class_balance


# def comprehensive_quality_assessment_with_csv(real_dir, synthetic_dir, output_dir="quality_reports", 
#                                                device='cuda', batch_size=32, 
#                                                fid_threshold=50.0, run_name=None):
#     """
#     Comprehensive quality assessment function that evaluates synthetic medical images
#     and saves all critical information to CSV files with timestamps.
    
#     Args:
#         real_dir: Directory containing real images
#         synthetic_dir: Directory containing synthetic images  
#         output_dir: Directory to save CSV reports
#         device: Computing device ('cuda' or 'cpu')
#         batch_size: Batch size for processing
#         fid_threshold: Maximum allowed FID score (lower is better)
#         run_name: Optional name for the evaluation run
        
#     Returns:
#         dict: Complete assessment results with paths to generated CSV files
        
#     Example:
#         results = comprehensive_quality_assessment_with_csv(
#             real_dir="C:/data/train",
#             synthetic_dir="C:/data/synth/train",
#             output_dir="quality_reports",
#             fid_threshold=50.0,
#             run_name="experiment_1"
#         )
#     """
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Generate timestamp and run identifier
#     timestamp = datetime.now()
#     timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
#     run_id = run_name if run_name else f"gan_eval_{timestamp_str}"
    
#     print(f"🔍 Starting comprehensive quality assessment: {run_id}")
#     print(f"📅 Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"📁 Output directory: {output_dir}")
    
#     # Initialize results dictionary
#     assessment_results = {
#         'run_info': {
#             'run_id': run_id,
#             'timestamp': timestamp.isoformat(),
#             'real_dir': str(real_dir),
#             'synthetic_dir': str(synthetic_dir),
#             'fid_threshold': fid_threshold,
#             'device': device,
#             'batch_size': batch_size
#         },
#         'class_metrics': {},
#         'overall_metrics': {},
#         'filtering_results': {},
#         'data_balance': {}
#     }
    
#     try:
#         # Step 1: Load Inception model for FID calculation
#         print("\n1️⃣ Loading Inception model...")
#         inception_model = load_inception_model(device)
        
#         # Step 2: Evaluate synthetic quality (using existing function)
#         print("\n2️⃣ Evaluating synthetic image quality...")
#         quality_scores = evaluate_synthetic_quality(real_dir, synthetic_dir, device, batch_size)
#         assessment_results['class_metrics'] = quality_scores
        
#         # Step 3: Assess class balance before filtering
#         print("\n3️⃣ Assessing class balance...")
#         real_balance = assess_class_balance(real_dir)
#         synth_balance_before = assess_class_balance(synthetic_dir)
        
#         assessment_results['data_balance']['real'] = real_balance
#         assessment_results['data_balance']['synthetic_before_filtering'] = synth_balance_before
        
#         # Step 4: Filter images based on quality with detailed tracking
#         print("\n4️⃣ Filtering low-quality images...")
#         filter_results = filter_synthetic_by_quality_with_tracking(
#             synthetic_dir, fid_threshold, real_dir, quality_scores
#         )
#         assessment_results['filtering_results'] = filter_results
        
#         # Step 5: Assess balance after filtering
#         print("\n5️⃣ Re-assessing balance after filtering...")
#         synth_balance_after = assess_class_balance(synthetic_dir)
#         assessment_results['data_balance']['synthetic_after_filtering'] = synth_balance_after
        
#         # Step 6: Calculate overall metrics
#         print("\n6️⃣ Calculating overall assessment metrics...")
#         overall_metrics = calculate_overall_metrics(assessment_results)
#         assessment_results['overall_metrics'] = overall_metrics
        
#         # Step 7: Generate comprehensive CSV reports
#         print("\n7️⃣ Generating detailed CSV reports...")
#         csv_files = generate_csv_reports(assessment_results, output_dir, timestamp_str)
#         assessment_results['report_files'] = csv_files
        
#         # Step 8: Save complete results as JSON for backup
#         json_file = os.path.join(output_dir, f"complete_assessment_{timestamp_str}.json")
#         with open(json_file, 'w') as f:
#             json.dump(assessment_results, f, indent=2, default=str)
        
#         assessment_results['json_backup'] = json_file
        
#         # Step 9: Generate summary report
#         print_assessment_summary(assessment_results)
        
#         print(f"\n✅ Assessment complete! Reports saved to: {output_dir}")
#         print(f"📊 Generated files:")
#         for file_path in csv_files + [json_file]:
#             print(f"   📄 {file_path}")
        
#         return assessment_results
        
#     except Exception as e:
#         print(f"❌ Error during assessment: {str(e)}")
#         # Still try to save partial results
#         assessment_results['error'] = str(e)
#         error_file = os.path.join(output_dir, f"error_report_{timestamp_str}.json")
#         with open(error_file, 'w') as f:
#             json.dump(assessment_results, f, indent=2, default=str)
        
#         raise


# def filter_synthetic_by_quality_with_tracking(synth_dir, fid_threshold=50.0, real_dir=None, quality_scores=None):
#     """
#     Enhanced filtering function that tracks detailed statistics about removed images.
    
#     Args:
#         synth_dir: Directory containing synthetic images
#         fid_threshold: Maximum allowed FID score
#         real_dir: Directory containing real images (optional)
#         quality_scores: Pre-computed quality scores (optional)
    
#     Returns:
#         dict: Detailed filtering statistics including per-class breakdown
#     """
#     print(f"🔍 Filtering synthetic images with FID threshold: {fid_threshold}")
    
#     filter_stats = {
#         'threshold_used': fid_threshold,
#         'filtering_method': 'FID-based' if real_dir else 'Statistical',
#         'classes_processed': {},
#         'total_before': 0,
#         'total_after': 0,
#         'total_removed': 0,
#         'removal_rate': 0.0
#     }
    
#     if real_dir is None or quality_scores is None:
#         # Basic statistical filtering fallback
#         print("⚠️  Warning: No FID scores available. Using basic statistical filtering...")
#         filter_stats['filtering_method'] = 'Brightness_Statistical'
        
#         for cls in os.listdir(synth_dir):
#             cls_path = os.path.join(synth_dir, cls)
#             if not os.path.isdir(cls_path):
#                 continue
            
#             images = glob.glob(os.path.join(cls_path, "*.png"))
#             if not images:
#                 images = glob.glob(os.path.join(cls_path, "*.jpg"))
            
#             filter_stats['classes_processed'][cls] = {
#                 'before': len(images),
#                 'removed': 0,
#                 'after': len(images),
#                 'method': 'brightness_check',
#                 'fid_score': 'N/A'
#             }
#             filter_stats['total_before'] += len(images)
            
#             # Simple quality check: remove very dark or very bright images
#             removed_count = 0
#             for img_path in images:
#                 try:
#                     img = Image.open(img_path).convert('L')
#                     img_array = np.array(img)
#                     mean_brightness = np.mean(img_array)
                    
#                     # Remove extreme brightness images (likely artifacts)
#                     if mean_brightness < 10 or mean_brightness > 245:
#                         os.remove(img_path)
#                         removed_count += 1
                        
#                 except Exception as e:
#                     print(f"⚠️  Error processing {img_path}: {e}")
#                     os.remove(img_path)
#                     removed_count += 1
            
#             filter_stats['classes_processed'][cls]['removed'] = removed_count
#             filter_stats['classes_processed'][cls]['after'] = len(images) - removed_count
#             filter_stats['total_removed'] += removed_count
            
#             if removed_count > 0:
#                 print(f"   🗑️  {cls}: Removed {removed_count}/{len(images)} images")
#             else:
#                 print(f"   ✅ {cls}: All {len(images)} images passed basic checks")
                
#     else:
#         # FID-based filtering with detailed tracking
#         print("📊 Using FID-based filtering...")
        
#         for cls, scores in quality_scores.items():
#             fid_score = scores['fid']
#             cls_path = os.path.join(synth_dir, cls)
#             images = glob.glob(os.path.join(cls_path, "*.png"))
#             if not images:
#                 images = glob.glob(os.path.join(cls_path, "*.jpg"))
            
#             filter_stats['classes_processed'][cls] = {
#                 'before': len(images),
#                 'fid_score': fid_score,
#                 'method': 'FID_threshold',
#                 'threshold_met': fid_score <= fid_threshold
#             }
#             filter_stats['total_before'] += len(images)
            
#             if fid_score > fid_threshold:
#                 # Remove all synthetic images from this class (poor quality)
#                 removed_count = 0
#                 for img_path in images:
#                     try:
#                         os.remove(img_path)
#                         removed_count += 1
#                     except Exception as e:
#                         print(f"⚠️  Error removing {img_path}: {e}")
                
#                 filter_stats['classes_processed'][cls].update({
#                     'removed': removed_count,
#                     'after': 0,
#                     'action': 'removed_all'
#                 })
#                 filter_stats['total_removed'] += removed_count
                
#                 print(f"   🗑️  {cls}: Removed all {removed_count} images (FID: {fid_score:.3f} > {fid_threshold})")
                
#             else:
#                 filter_stats['classes_processed'][cls].update({
#                     'removed': 0,
#                     'after': len(images),
#                     'action': 'kept_all'
#                 })
#                 print(f"   ✅ {cls}: Kept all {len(images)} images (FID: {fid_score:.3f} ≤ {fid_threshold})")
    
#     # Calculate final statistics
#     filter_stats['total_after'] = filter_stats['total_before'] - filter_stats['total_removed']
#     filter_stats['removal_rate'] = (filter_stats['total_removed'] / filter_stats['total_before'] * 100 
#                                    if filter_stats['total_before'] > 0 else 0)
    
#     print(f"🏁 Filtering complete: Removed {filter_stats['total_removed']}/{filter_stats['total_before']} "
#           f"images ({filter_stats['removal_rate']:.1f}% removal rate)")
    
#     return filter_stats


# def calculate_overall_metrics(assessment_results):
#     """
#     Calculate comprehensive overall assessment metrics from detailed results.
    
#     Args:
#         assessment_results: Dictionary containing all assessment data
        
#     Returns:
#         dict: Overall quality metrics and statistics
#     """
#     class_metrics = assessment_results['class_metrics']
#     filter_results = assessment_results['filtering_results']
    
#     if not class_metrics:
#         return {
#             'status': 'no_classes_evaluated',
#             'message': 'No classes could be evaluated',
#             'quality_grade': 'Unacceptable'
#         }
    
#     # FID statistics
#     fid_scores = [scores['fid'] for scores in class_metrics.values() if 'fid' in scores]
    
#     if not fid_scores:
#         return {
#             'status': 'no_fid_scores',
#             'message': 'No FID scores available',
#             'quality_grade': 'Unacceptable'
#         }
    
#     # Data statistics
#     total_real = sum(scores.get('real_count', 0) for scores in class_metrics.values())
#     total_synth_before = assessment_results['data_balance'].get('synthetic_before_filtering', {}).get('total_images', 0)
#     total_synth_after = filter_results.get('total_after', 0)
    
#     # Calculate quality grades and statistics
#     fid_mean = np.mean(fid_scores)
#     fid_std = np.std(fid_scores) if len(fid_scores) > 1 else 0
#     quality_grade = get_quality_grade(fid_mean)
    
#     # Calculate pass rate
#     threshold = assessment_results['run_info']['fid_threshold']
#     classes_passed = len([s for s in fid_scores if s <= threshold])
#     pass_rate = classes_passed / len(fid_scores) * 100 if fid_scores else 0
    
#     # Calculate balance score
#     balance_score = calculate_balance_score(assessment_results['data_balance'])
    
#     overall_metrics = {
#         # Status
#         'status': 'completed',
#         'quality_grade': quality_grade,
        
#         # FID Analysis
#         'fid_mean': float(fid_mean),
#         'fid_std': float(fid_std),
#         'fid_min': float(min(fid_scores)),
#         'fid_max': float(max(fid_scores)),
#         'fid_median': float(np.median(fid_scores)),
        
#         # Class Performance
#         'classes_evaluated': len(class_metrics),
#         'classes_passed_threshold': int(classes_passed),
#         'pass_rate': float(pass_rate),
        
#         # Data Statistics
#         'total_real_images': int(total_real),
#         'total_synthetic_before': int(total_synth_before),
#         'total_synthetic_after': int(total_synth_after),
#         'images_removed': int(filter_results.get('total_removed', 0)),
#         'synthetic_retention_rate': float(total_synth_after / total_synth_before * 100 if total_synth_before > 0 else 0),
        
#         # Ratios
#         'synthetic_to_real_ratio_before': float(total_synth_before / total_real if total_real > 0 else 0),
#         'synthetic_to_real_ratio_after': float(total_synth_after / total_real if total_real > 0 else 0),
        
#         # Balance and Quality
#         'data_balance_score': float(balance_score),
#         'overall_score': calculate_overall_score(fid_mean, pass_rate, balance_score)
#     }
    
#     return overall_metrics


# def calculate_balance_score(balance_data):
#     """
#     Calculate a balance score based on class distribution uniformity.
#     Returns a score from 0-100 where 100 is perfect balance.
#     """
#     try:
#         synth_after = balance_data.get('synthetic_after_filtering', {})
#         if not synth_after or 'class_counts' not in synth_after:
#             return 0.0
        
#         counts = list(synth_after['class_counts'].values())
#         counts = [c for c in counts if c > 0]  # Remove empty classes
        
#         if len(counts) <= 1:
#             return 100.0 if len(counts) == 1 else 0.0
        
#         # Calculate coefficient of variation (lower = better balance)
#         mean_count = np.mean(counts)
#         std_count = np.std(counts)
#         cv = std_count / mean_count if mean_count > 0 else float('inf')
        
#         # Convert to 0-100 scale (100 = perfect balance)
#         balance_score = max(0, 100 - (cv * 50))  # Scale factor for medical data
#         return float(balance_score)
        
#     except Exception as e:
#         print(f"⚠️  Error calculating balance score: {e}")
#         return 0.0


# def calculate_overall_score(fid_mean, pass_rate, balance_score):
#     """
#     Calculate an overall quality score combining FID, pass rate, and balance.
#     Returns a score from 0-100.
#     """
#     try:
#         # Normalize FID (lower is better, typical range 0-200)
#         fid_score = max(0, 100 - (fid_mean / 2))  
        
#         # Weight the components
#         overall_score = (fid_score * 0.5) + (pass_rate * 0.3) + (balance_score * 0.2)
        
#         return float(min(100, max(0, overall_score)))
        
#     except Exception:
#         return 0.0


# def get_quality_grade(fid_score):
#     """
#     Assign quality grade based on FID score for medical images.
#     Based on medical imaging literature and clinical requirements.
#     """
#     if fid_score <= 20:
#         return "Excellent"
#     elif fid_score <= 40:
#         return "Good" 
#     elif fid_score <= 60:
#         return "Fair"
#     elif fid_score <= 100:
#         return "Poor"
#     else:
#         return "Unacceptable"


# def print_assessment_summary(assessment_results):
#     """
#     Print a formatted summary of the assessment results.
#     """
#     print("\n" + "="*60)
#     print("📊 QUALITY ASSESSMENT SUMMARY")
#     print("="*60)
    
#     overall = assessment_results.get('overall_metrics', {})
#     run_info = assessment_results['run_info']
    
#     print(f"🏷️  Run ID: {run_info['run_id']}")
#     print(f"📅 Date: {datetime.fromisoformat(run_info['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"🎯 Overall Grade: {overall.get('quality_grade', 'N/A')}")
#     print(f"📈 Overall Score: {overall.get('overall_score', 0):.1f}/100")
    
#     print(f"\n🔍 FID ANALYSIS:")
#     print(f"   Mean FID: {overall.get('fid_mean', 0):.3f}")
#     print(f"   Best FID: {overall.get('fid_min', 0):.3f}")
#     print(f"   Worst FID: {overall.get('fid_max', 0):.3f}")
#     print(f"   Pass Rate: {overall.get('pass_rate', 0):.1f}% ({overall.get('classes_passed_threshold', 0)}/{overall.get('classes_evaluated', 0)} classes)")
    
#     print(f"\n📁 DATA STATISTICS:")
#     print(f"   Real Images: {overall.get('total_real_images', 0):,}")
#     print(f"   Synthetic Before: {overall.get('total_synthetic_before', 0):,}")
#     print(f"   Synthetic After: {overall.get('total_synthetic_after', 0):,}")
#     print(f"   Images Removed: {overall.get('images_removed', 0):,}")
#     print(f"   Retention Rate: {overall.get('synthetic_retention_rate', 0):.1f}%")
    
#     print(f"\n⚖️  BALANCE SCORE: {overall.get('data_balance_score', 0):.1f}/100")
#     print("="*60)


# def generate_csv_reports(assessment_results, output_dir, timestamp_str):
#     """
#     Generate comprehensive CSV reports from assessment results.
    
#     Args:
#         assessment_results: Dictionary containing all assessment data
#         output_dir: Directory to save CSV files
#         timestamp_str: Timestamp string for file naming
        
#     Returns:
#         list: Paths to generated CSV files
#     """
#     csv_files = []
    
#     try:
#         # 1. Summary Report CSV
#         summary_file = os.path.join(output_dir, f"summary_report_{timestamp_str}.csv")
#         summary_df = create_summary_report_df(assessment_results)
#         summary_df.to_csv(summary_file, index=False)
#         csv_files.append(summary_file)
        
#         # 2. Per-Class Detailed Report CSV  
#         class_detail_file = os.path.join(output_dir, f"class_details_{timestamp_str}.csv")
#         class_df = create_class_details_df(assessment_results)
#         class_df.to_csv(class_detail_file, index=False)
#         csv_files.append(class_detail_file)
        
#         # 3. Filtering Results CSV
#         filtering_file = os.path.join(output_dir, f"filtering_results_{timestamp_str}.csv")
#         filtering_df = create_filtering_results_df(assessment_results)
#         filtering_df.to_csv(filtering_file, index=False)
#         csv_files.append(filtering_file)
        
#         # 4. Data Balance Analysis CSV
#         balance_file = os.path.join(output_dir, f"data_balance_{timestamp_str}.csv")
#         balance_df = create_balance_analysis_df(assessment_results)
#         balance_df.to_csv(balance_file, index=False)
#         csv_files.append(balance_file)
        
#         # 5. Complete Metrics CSV (all metrics in one file)
#         complete_file = os.path.join(output_dir, f"complete_metrics_{timestamp_str}.csv")
#         complete_df = create_complete_metrics_df(assessment_results)
#         complete_df.to_csv(complete_file, index=False)
#         csv_files.append(complete_file)
        
#         print(f"✅ Successfully generated {len(csv_files)} CSV reports")
        
#     except Exception as e:
#         print(f"❌ Error generating CSV reports: {str(e)}")
        
#     return csv_files


# def create_summary_report_df(assessment_results):
#     """Create high-level summary report DataFrame."""
    
#     run_info = assessment_results['run_info']
#     overall = assessment_results.get('overall_metrics', {})
#     filter_results = assessment_results.get('filtering_results', {})
    
#     # Create single-row summary with all key metrics
#     summary_data = {
#         'Run_ID': [run_info.get('run_id', 'unknown')],
#         'Timestamp': [run_info.get('timestamp', '')],
#         'Date': [datetime.fromisoformat(run_info.get('timestamp', datetime.now().isoformat())).strftime('%Y-%m-%d')],
#         'Time': [datetime.fromisoformat(run_info.get('timestamp', datetime.now().isoformat())).strftime('%H:%M:%S')],
        
#         # Quality Assessment
#         'Overall_Quality_Grade': [overall.get('quality_grade', 'N/A')],
#         'Overall_Score': [round(overall.get('overall_score', 0), 1)],
#         'Mean_FID_Score': [round(overall.get('fid_mean', 0), 3)],
#         'FID_Std_Dev': [round(overall.get('fid_std', 0), 3)],
#         'Min_FID_Score': [round(overall.get('fid_min', 0), 3)],
#         'Max_FID_Score': [round(overall.get('fid_max', 0), 3)],
#         'Median_FID_Score': [round(overall.get('fid_median', 0), 3)],
        
#         # Class Performance
#         'Classes_Evaluated': [overall.get('classes_evaluated', 0)],
#         'Classes_Passed_Threshold': [overall.get('classes_passed_threshold', 0)],
#         'Pass_Rate_Percent': [round(overall.get('pass_rate', 0), 1)],
        
#         # Data Statistics
#         'Total_Real_Images': [overall.get('total_real_images', 0)],
#         'Total_Synthetic_Before_Filter': [overall.get('total_synthetic_before', 0)],
#         'Total_Synthetic_After_Filter': [overall.get('total_synthetic_after', 0)],
#         'Images_Removed': [filter_results.get('total_removed', 0)],
#         'Removal_Rate_Percent': [round(filter_results.get('removal_rate', 0), 1)],
#         'Retention_Rate_Percent': [round(overall.get('synthetic_retention_rate', 0), 1)],
        
#         # Ratios and Balance
#         'Synthetic_to_Real_Ratio_Before': [round(overall.get('synthetic_to_real_ratio_before', 0), 2)],
#         'Synthetic_to_Real_Ratio_After': [round(overall.get('synthetic_to_real_ratio_after', 0), 2)],
#         'Data_Balance_Score': [round(overall.get('data_balance_score', 0), 1)],
        
#         # Configuration
#         'FID_Threshold_Used': [run_info.get('fid_threshold', 0)],
#         'Device_Used': [run_info.get('device', 'unknown')],
#         'Batch_Size': [run_info.get('batch_size', 0)],
#         'Filter_Method': [filter_results.get('filtering_method', 'unknown')],
        
#         # Paths
#         'Real_Data_Path': [run_info.get('real_dir', '')],
#         'Synthetic_Data_Path': [run_info.get('synthetic_dir', '')]
#     }
    
#     return pd.DataFrame(summary_data)


# def create_class_details_df(assessment_results):
#     """Create detailed per-class metrics DataFrame."""
    
#     class_metrics = assessment_results.get('class_metrics', {})
#     filter_results = assessment_results.get('filtering_results', {})
#     run_info = assessment_results['run_info']
    
#     class_data = []
    
#     for class_name, metrics in class_metrics.items():
#         filter_info = filter_results.get('classes_processed', {}).get(class_name, {})
#         fid_score = metrics.get('fid', float('inf'))
        
#         class_row = {
#             'Run_ID': run_info.get('run_id', 'unknown'),
#             'Class_Name': class_name,
#             'FID_Score': round(fid_score, 3),
#             'Quality_Grade': get_quality_grade(fid_score),
#             'Real_Images_Count': metrics.get('real_count', 0),
#             'Synthetic_Images_Before': metrics.get('synth_count', 0),
#             'Images_Removed': filter_info.get('removed', 0),
#             'Images_After_Filter': filter_info.get('after', 0),
#             'Passed_Threshold': fid_score <= run_info.get('fid_threshold', 50),
#             'Retention_Rate_Percent': round(
#                 (filter_info.get('after', 0) / metrics.get('synth_count', 1) * 100) 
#                 if metrics.get('synth_count', 0) > 0 else 0, 1
#             ),
#             'Synthetic_to_Real_Ratio_Before': round(
#                 metrics.get('synth_count', 0) / metrics.get('real_count', 1)
#                 if metrics.get('real_count', 0) > 0 else 0, 2
#             ),
#             'Synthetic_to_Real_Ratio_After': round(
#                 filter_info.get('after', 0) / metrics.get('real_count', 1)
#                 if metrics.get('real_count', 0) > 0 else 0, 2
#             ),
#             'Filter_Action': filter_info.get('action', 'unknown'),
#             'Filter_Method': filter_info.get('method', 'unknown')
#         }
        
#         class_data.append(class_row)
    
#     return pd.DataFrame(class_data)


# def create_filtering_results_df(assessment_results):
#     """Create filtering results and statistics DataFrame."""
    
#     filter_results = assessment_results.get('filtering_results', {})
#     run_info = assessment_results['run_info']
    
#     # Overall filtering statistics
#     overall_row = {
#         'Run_ID': run_info.get('run_id', 'unknown'),
#         'Category': 'Overall',
#         'Class_Name': 'ALL_CLASSES',
#         'Images_Before': filter_results.get('total_before', 0),
#         'Images_Removed': filter_results.get('total_removed', 0),
#         'Images_After': filter_results.get('total_after', 0),
#         'Removal_Rate_Percent': round(filter_results.get('removal_rate', 0), 1),
#         'Filter_Method': filter_results.get('filtering_method', 'unknown'),
#         'Threshold_Used': filter_results.get('threshold_used', 0),
#         'Action_Taken': 'FILTERED' if filter_results.get('total_removed', 0) > 0 else 'NO_ACTION'
#     }
    
#     filtering_data = [overall_row]
    
#     # Per-class filtering statistics
#     for class_name, class_filter in filter_results.get('classes_processed', {}).items():
#         class_row = {
#             'Run_ID': run_info.get('run_id', 'unknown'),
#             'Category': 'Class_Detail',
#             'Class_Name': class_name,
#             'Images_Before': class_filter.get('before', 0),
#             'Images_Removed': class_filter.get('removed', 0),
#             'Images_After': class_filter.get('after', 0),
#             'Removal_Rate_Percent': round(
#                 (class_filter.get('removed', 0) / class_filter.get('before', 1) * 100)
#                 if class_filter.get('before', 0) > 0 else 0, 1
#             ),
#             'Filter_Method': class_filter.get('method', 'unknown'),
#             'Threshold_Used': filter_results.get('threshold_used', 0),
#             'Action_Taken': class_filter.get('action', 'unknown'),
#             'FID_Score': class_filter.get('fid_score', 'N/A'),
#             'Threshold_Met': class_filter.get('threshold_met', False)
#         }
        
#         filtering_data.append(class_row)
    
#     return pd.DataFrame(filtering_data)


# def create_balance_analysis_df(assessment_results):
#     """Create comprehensive data balance analysis DataFrame."""
    
#     balance_data = assessment_results.get('data_balance', {})
#     run_info = assessment_results['run_info']
    
#     balance_rows = []
    
#     # Process each balance analysis stage
#     for stage_name, stage_data in balance_data.items():
#         if not stage_data or 'class_counts' not in stage_data:
#             continue
            
#         stage_display = {
#             'real': 'Real_Data',
#             'synthetic_before_filtering': 'Synthetic_Before_Filter', 
#             'synthetic_after_filtering': 'Synthetic_After_Filter'
#         }.get(stage_name, stage_name)
        
#         # Overall statistics for this stage
#         overall_row = {
#             'Run_ID': run_info.get('run_id', 'unknown'),
#             'Data_Stage': stage_display,
#             'Analysis_Type': 'Overall_Statistics',
#             'Class_Name': 'ALL_CLASSES',
#             'Image_Count': stage_data.get('total_images', 0),
#             'Min_Count': stage_data.get('min_count', 0),
#             'Max_Count': stage_data.get('max_count', 0),
#             'Imbalance_Ratio': round(stage_data.get('imbalance_ratio', 0), 2),
#             'Std_Deviation': round(stage_data.get('std_dev', 0), 2),
#             'Unique_Classes': len(stage_data.get('class_counts', {}))
#         }
#         balance_rows.append(overall_row)
        
#         # Per-class counts and statistics
#         for class_name, count in stage_data.get('class_counts', {}).items():
#             class_row = {
#                 'Run_ID': run_info.get('run_id', 'unknown'),
#                 'Data_Stage': stage_display,
#                 'Analysis_Type': 'Class_Detail',
#                 'Class_Name': class_name,
#                 'Image_Count': count,
#                 'Percentage_of_Total': round(
#                     (count / stage_data.get('total_images', 1) * 100)
#                     if stage_data.get('total_images', 0) > 0 else 0, 1
#                 ),
#                 'Min_Count': stage_data.get('min_count', 0),
#                 'Max_Count': stage_data.get('max_count', 0),
#                 'Is_Min_Class': count == stage_data.get('min_count', 0),
#                 'Is_Max_Class': count == stage_data.get('max_count', 0)
#             }
#             balance_rows.append(class_row)
    
#     return pd.DataFrame(balance_rows)


# def create_complete_metrics_df(assessment_results):
#     """Create complete metrics DataFrame with all information consolidated."""
    
#     run_info = assessment_results['run_info']
#     overall = assessment_results.get('overall_metrics', {})
    
#     # Create comprehensive single-row record
#     complete_data = []
    
#     # Basic run information
#     base_row = {
#         'Run_ID': run_info.get('run_id', 'unknown'),
#         'Timestamp': run_info.get('timestamp', ''),
#         'Date': datetime.fromisoformat(run_info.get('timestamp', datetime.now().isoformat())).strftime('%Y-%m-%d'),
#         'Time': datetime.fromisoformat(run_info.get('timestamp', datetime.now().isoformat())).strftime('%H:%M:%S'),
#         'Real_Data_Path': run_info.get('real_dir', ''),
#         'Synthetic_Data_Path': run_info.get('synthetic_dir', ''),
#         'FID_Threshold': run_info.get('fid_threshold', 0),
#         'Device_Used': run_info.get('device', 'unknown'),
#         'Batch_Size': run_info.get('batch_size', 0)
#     }
    
#     # Add all overall metrics with prefix
#     for key, value in overall.items():
#         if isinstance(value, (int, float)):
#             base_row[f'Overall_{key}'] = value
#         else:
#             base_row[f'Overall_{key}'] = str(value)
    
#     # Add per-class metrics
#     for class_name, metrics in assessment_results.get('class_metrics', {}).items():
#         safe_class_name = class_name.replace(' ', '_').replace('-', '_')
#         base_row[f'Class_{safe_class_name}_FID'] = round(metrics.get('fid', 0), 3)
#         base_row[f'Class_{safe_class_name}_Real_Count'] = metrics.get('real_count', 0)
#         base_row[f'Class_{safe_class_name}_Synth_Count'] = metrics.get('synth_count', 0)
#         base_row[f'Class_{safe_class_name}_Quality_Grade'] = get_quality_grade(metrics.get('fid', float('inf')))
    
#     # Add filtering results
#     filter_results = assessment_results.get('filtering_results', {})
#     base_row.update({
#         'Filter_Method': filter_results.get('filtering_method', 'unknown'),
#         'Total_Images_Removed': filter_results.get('total_removed', 0),
#         'Removal_Rate_Percent': round(filter_results.get('removal_rate', 0), 1)
#     })
    
#     complete_data.append(base_row)
    
#     return pd.DataFrame(complete_data)



from pathlib import Path
DATA_ROOT = Path(r"C:data")              # Root directory with train/valid/test folders
SYNTH_ROOT = Path(r"C:data\synth\train")   # Where synthetic images will be saved
MERGED_ROOT = Path(r"C:data\merged_train") # Where real+synthetic merged data will be stored
REAL_ROOT = Path(r"C:\Users\sapounaki.m\Desktop\2D_CANCER\data\train")

if __name__ == "__main__":
    # Get balance before filtering
    balance_before = assess_class_balance(SYNTH_ROOT)

    # Execute the complete
    filter_synthetic_by_quality(SYNTH_ROOT, fid_threshold=50.0, real_dir=os.path.join(DATA_ROOT, "train"))

  # Get balance after filtering  
    balance_after = assess_class_balance(SYNTH_ROOT)
    
    # Get quality scores
    quality_scores = evaluate_synthetic_quality(
        real_dir=os.path.join(DATA_ROOT, "train"),
        synthetic_dir=SYNTH_ROOT
    )
 
    # Save CSV report with balance info
    csv_file = save_quality_report_csv(
        quality_scores, 
        num_of_epch=4, 
        num_of_iters=3,
        balance_before=balance_before,
        balance_after=balance_after,
        output_dir="quality_reports", 
        run_name="medical_gan_v1"
    )


