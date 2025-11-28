# Main execution script for medical image data augmentation using GANs
# This script orchestrates the complete pipeline: baseline training -> GAN training -> 
# synthetic data generation -> augmented training -> performance comparison

import os, torch
from data import build_loaders, count_classes
from pathlib import Path
from classifier import make_classifier, train_classifier
from train_gan import train_gan
from quality_utils import filter_synthetic_by_quality, assess_class_balance, evaluate_synthetic_quality, save_quality_report_csv
from sample_gan import sample_to_folder
from merge_data import merge_real_synth
from resume_training import resume_train_gan

# Data directory paths - adjust these for your specific setup
DATA_ROOT = Path(r"C:data")              # Root directory with train/valid/test folders
SYNTH_ROOT = Path(r"C:data\synth\train")   # Where synthetic images will be saved
MERGED_ROOT = Path(r"C:data\merged_train") # Where real+synthetic merged data will be stored

EPOCHS = 120
ITERS = 40000
DEVICE = "cuda"
z_dim = 128

# C:\Users\sapounaki.m\Desktop\2D_CANCER\data\train
def main():
    """
    Execute the complete medical image data augmentation pipeline.
    
    This function demonstrates the effectiveness osf GAN-based data augmentation
    for medical image classification by comparing baseline performance against
    augmented dataset performance.
    """

    count_classes(data_root=os.path.join(DATA_ROOT, "train"))
    names_of_classes = sorted(os.listdir(os.path.join(DATA_ROOT, "train")))
    
    # === STEP 1: BASELINE CLASSIFIER TRAINING ===
    # Train a classifier on original real medical images only
    # This establishes the baseline performance before data augmentation
    print("Step 1: Training baseline classifier on real data...")

    # Load the original medical image datasets with proper transforms
    train_dl, val_dl, test_dl, classes = build_loaders(DATA_ROOT, batch_size=32)
    
    # Create classifier architecture adapted for medical images
    clf = make_classifier(num_classes=len(classes), device=DEVICE)
    

    # Train baseline classifier using only real medical images
    # This typically suffers from limited data, especially for rare cancer types
    clf = train_classifier(clf, train_dl, val_dl, epochs=EPOCHS, lr=1e-3, device=DEVICE)
    
    # === STEP 2: GAN TRAINING ===
    # Train a conditional GAN to learn the distribution of real medical images
    # The GAN will learn to generate synthetic images for each cancer type
    print("Step 2: Training conditional GAN on real training data...")

    
    # Train GAN using the same real training data
    # Returns EMA generator for stable, high-quality synthetic image generation
    # When training from scratch run, g_ema = train_gan(train_dl, num_classes=4, iters=20000, device=DEVICE)
    # When resuming from checkpoint, use g_ema = train_gan(train_dl, num_classes=4, iters=40000, device=DEVICE, checkpoint_path="checkpoints/gan_020000.pt")   checkpoint_path="checkpoints/gan_020000.pt")

    g_ema = train_gan(train_dl, num_classes=4, iters=ITERS,  save_interval=1000, device=DEVICE, checkpoint_path="checkpoints/gan_038000.pt") 
    print("Resuming from checkpoint...")
    # g_ema = train_gan(train_dl, num_classes=len(classes), save_interval=1, iters=ITERS, device=DEVICE)

    # === STEP 3: SYNTHETIC DATA GENERATION ===
    # Generate synthetic medical images to augment the training dataset
    # This addresses data scarcity issues common in medical imaging
    print("Step 3: Generating synthetic medical images...")
    
    # Ensure synthetic data directory exists
    if os.path.exists(SYNTH_ROOT) is False:
        os.makedirs(SYNTH_ROOT, exist_ok=True)
    
    # Generate balanced synthetic dataset: 1000 images per cancer type
    # This ensures each class has sufficient representation for training
    sample_to_folder(g_ema, SYNTH_ROOT, per_class=100, num_classes=len(classes), names_of_classes=names_of_classes, device=DEVICE)


    # === NEW STEP 3.5: QUALITY FILTERING ===
    print("Step 3.5: Filtering low-quality synthetic images...")

    # Get balance before filtering
    balance_before = assess_class_balance(SYNTH_ROOT)

    # Execute the complete
    # filter_synthetic_by_quality(SYNTH_ROOT, fid_threshold=50.0, real_dir=os.path.join(DATA_ROOT, "train"))

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
        num_of_epch=EPOCHS, 
        num_of_iters=ITERS,
        balance_before=balance_before,
        balance_after=balance_after,
        output_dir="quality_reports", 
        run_name="medical_gan_v1"
    )


    # === STEP 4: DATA MERGING ===
    # Combine real and synthetic images into augmented training dataset
    # This preserves original medical images while adding synthetic diversity
    print("Step 4: Merging real and synthetic data...")
    
    merge_real_synth(os.path.join(DATA_ROOT, "train"), SYNTH_ROOT, MERGED_ROOT)
    
    # Note: We reuse validation/test sets from real-only data to ensure fair comparison
    # Synthetic data is only used for training augmentation
    
    # Build data loader for the merged (real + synthetic) training dataset
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from data import build_transforms
    
    # Create dataset from merged directory with training transforms
    merged_ds = datasets.ImageFolder(MERGED_ROOT, transform=build_transforms(True))
    merged_dl = DataLoader(merged_ds, batch_size=32, shuffle=True, num_workers=4)

    # === STEP 5: AUGMENTED CLASSIFIER TRAINING ===
    # Train a new classifier on the augmented (real + synthetic) dataset
    # This should show improved performance due to increased data diversity
    print("Step 5: Training classifier on augmented data...")
    
    # Create fresh classifier with same architecture as baseline
    clf_aug = make_classifier(num_classes=len(classes), device=DEVICE)
    
    # Train on merged dataset: real images + GAN-generated synthetic images
    # Uses same validation set for fair comparison with baseline
    clf_aug = train_classifier(clf_aug, merged_dl, val_dl, epochs=100, lr=1e-3, device=DEVICE)

    # === STEP 6: PERFORMANCE EVALUATION ===  
    # Compare baseline vs. augmented classifier performance on same test set
    # This demonstrates the effectiveness of GAN-based data augmentation
    print("Step 6: Evaluating and comparing model performance...")

    
    def eval_acc(model, dl):
        """
        Evaluate model accuracy on a given dataset.
        
        Args:
            model: Trained classifier to evaluate
            dl: DataLoader with test images and labels
            
        Returns:
            float: Classification accuracy (0.0 to 1.0)
        """
        model.eval()  # Set to evaluation mode (disable dropout, etc.)
        device = "cuda"
        correct, total = 0, 0
        
        # Disable gradient computation for evaluation (saves memory)
        with torch.no_grad():
            for x, y in dl:
                # Move data to device
                x, y = x.to(device), y.to(device)
                
                # Get predictions (class with highest probability)
                pred = model(x).argmax(1)
                
                # Count correct predictions
                correct += (pred == y).sum().item()
                total += y.numel()  # Total number of samples
                
        return correct / total

    # Evaluate both classifiers on the same test set for fair comparison
    acc_real = eval_acc(clf, test_dl)      # Baseline: trained on real data only
    acc_aug = eval_acc(clf_aug, test_dl)   # Augmented: trained on real + synthetic
    
    # Display results
    print("\n=== RESULTS ===")
    print(f"Test accuracy (baseline - real data only): {acc_real:.4f}")
    print(f"Test accuracy (augmented - real + synthetic): {acc_aug:.4f}")
    print(f"Improvement: {acc_aug - acc_real:.4f} ({((acc_aug - acc_real)/acc_real)*100:.2f}%)")
    
    # The augmented model should show improved performance, demonstrating
    # the effectiveness of GAN-based data augmentation for medical imaging

if __name__ == "__main__":
    # Execute the complete pipeline when script is run directly
    assert torch.cuda.is_available(), "CUDA is not available – aborting"
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)}")
    main()

