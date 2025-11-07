# Create a simpler inference-only script
import os
import torch
from pathlib import Path
from gan import Generator
from sample_gan import sample_to_folder
from quality_utils import filter_synthetic_by_quality, assess_class_balance, evaluate_synthetic_quality, save_quality_report_csv

def load_generator_from_checkpoint(checkpoint_path, num_classes=4, z_dim=128, device="cuda"):
    """
    Load the EMA generator from a checkpoint for inference.
    
    Args:
        checkpoint_path (str): Path to checkpoint.pt file
        num_classes (int): Number of classes (must match training)
        z_dim (int): Latent dimension (must match training)
        device (str): Device to load model on
        
    Returns:
        Generator: The loaded EMA generator ready for inference
    """
    device = torch.device(device)
    
    # Create generator with same architecture as training
    g_ema = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load EMA weights (preferred for best quality)
    if "EMA" in checkpoint:
        g_ema.load_state_dict(checkpoint["EMA"], strict=False)
        print("✓ EMA generator loaded successfully")
    elif "G" in checkpoint:
        # Fallback to regular generator if EMA not available
        g_ema.load_state_dict(checkpoint["G"])
        print("⚠ EMA not found, loaded regular generator weights")
    else:
        raise ValueError("Checkpoint does not contain 'G' or 'EMA' keys!")
    
    # Set to evaluation mode
    g_ema.eval()
    
    return g_ema


def generate_samples(g_ema, num_classes=4, z_dim=128, device="cuda"):
    """
    Generate a few sample images for quick testing.
    
    Args:
        g_ema: The loaded generator
        num_classes (int): Number of classes
        z_dim (int): Latent dimension
        device (str): Device
    """
    from torchvision.utils import save_image
    import torch
    
    device = torch.device(device)
    g_ema.eval()
    
    with torch.no_grad():
        # Generate 8 samples per class
        z = torch.randn(num_classes * 8, z_dim, device=device)
        y = torch.arange(num_classes, device=device).repeat_interleave(8)
        
        # Generate images
        imgs = g_ema(z, y)
        
        # Convert from [-1, 1] to [0, 1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0
        
        # Save grid
        from torchvision.utils import make_grid
        grid = make_grid(imgs, nrow=8)
        save_image(grid, "test_samples1.png")
        print("✓ Test samples saved to: test_samples.png")


if __name__ == "__main__":
    # Configuration
    CHECKPOINT_PATH = Path(r"C:\Users\sapounaki.m\Desktop\2D_CANCER\checkpoints\Resumegan_014900.pt")   # Path to your checkpoint file
    NUM_CLASSES = 4  
    Z_DIM = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint '{CHECKPOINT_PATH}' not found!")
        exit(1)
    
    # Load generator
    g_ema = load_generator_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=NUM_CLASSES,
        z_dim=Z_DIM,
        device=DEVICE
    )
    
    print("\\n" + "="*60)
    print("Generator loaded successfully!")
    print("="*60)
    
    # Option 1: Generate a few test samples
    print("\\nGenerating test samples...")
    generate_samples(g_ema, NUM_CLASSES, Z_DIM, DEVICE)
    
    # Option 2: Generate large dataset (uncomment to use)
    print("\\nTo generate a full synthetic dataset, uncomment the code below:")
    print("="*60)
 
    # Get class names from your data
    DATA_ROOT = Path(r"C:data/train")
    names_of_classes = sorted(os.listdir(DATA_ROOT))
    SYNTH_ROOT = Path(r"C:data/Inference_synthetic_images")
    
    # Generate synthetic images
    sample_to_folder(
        g_ema=g_ema,
        out_root= SYNTH_ROOT,
        per_class=10,  # Number of images per class
        z_dim=Z_DIM,
        num_classes=NUM_CLASSES,
        names_of_classes=names_of_classes,
        device=DEVICE
    )
    print("✓ Synthetic dataset generated!")


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
        real_dir = DATA_ROOT,
        synthetic_dir = SYNTH_ROOT
    )
 
    # Save CSV report with balance info
    csv_file = save_quality_report_csv(
        quality_scores, 
        num_of_epch=100, 
        num_of_iters=15000,
        balance_before=balance_before,
        balance_after=balance_after,
        output_dir="quality_reports", 
        run_name="medical_gan_v1"
    )


 

# with open("load_checkpoint_inference.py", "w") as f:
#     f.write(inference_script)

# print("✓ Inference script created: load_checkpoint_inference.py")
# print("\nThis script is for INFERENCE ONLY (no training)")
# print("Use it when you just want to generate images from your checkpoint")