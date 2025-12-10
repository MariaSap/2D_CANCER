import os
import torch
from torchvision.utils import save_image
from gan import Generator
from PIL import Image
import numpy as np

def sample_to_folder(gema, outroot, per_class=2000, z_dim=128, num_classes=4,
                     names_of_classes=None, device='cuda'):
    """
    Generate synthetic images from a trained GAN and save them organized by class.
    
    CRITICAL FIX: Explicit range conversion [-1, 1] → [0, 1] → [0, 255]
    This ensures consistency with training data normalization.
    
    Args:
        gema: Exponential Moving Average version of trained Generator
        outroot: Root directory where synthetic images will be saved
        per_class: Number of synthetic images to generate per class
        z_dim: Dimension of the noise vector (latent space dimension)
        num_classes: Number of cancer types to generate (default 4)
        names_of_classes: List of class names for folder naming
        device: Computing device (cuda/cpu)
    """
    gema.eval()
    os.makedirs(outroot, exist_ok=True)
    
    print("=" * 70)
    print("Generating synthetic medical images...")
    print("=" * 70)
    
    with torch.no_grad():
        for cls in range(num_classes):
            cls_name = names_of_classes[cls] if names_of_classes else f"class_{cls}"
            cls_dir = os.path.join(outroot, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            
            print(f"\nGenerating {per_class} images for class {cls_name}...")
            
            for i in range(per_class):
                # Generate noise and class label
                z = torch.randn(1, z_dim, device=device)
                y = torch.tensor([cls], device=device)
                
                # Generate image from GAN
                x = gema(z, y)  # Output range: [-1, 1]
                
                # ===== CRITICAL FIX: Explicit range conversion =====
                # Convert [-1, 1] → [0, 1]
                x = (x + 1) / 2
                x = x.clamp(0, 1)  # Safety clamp to ensure [0, 1]
                
                # Save as grayscale PNG
                # Convert to numpy and then to uint8 [0, 255]
                x_np = (x.squeeze().cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(x_np, mode='L')
                img.save(os.path.join(cls_dir, f'{i:06d}.png'))
                
                if (i + 1) % 500 == 0:
                    print(f"  Generated {i+1}/{per_class} images")
            
            print(f"  ✓ Completed class {cls_name}")
    
    print("\n" + "=" * 70)
    print("Image generation completed!")
    print(f"Synthetic images saved to: {outroot}")
    print("=" * 70)

def load_checkpoint_and_sample(checkpoint_path, outroot, z_dim=128, num_classes=4,
                               per_class=2000, names_of_classes=None, device='cuda'):
    """
    Load a trained GAN checkpoint and generate synthetic images.
    
    Args:
        checkpoint_path: Path to saved checkpoint file
        outroot: Directory to save generated images
        z_dim: Noise dimension (must match checkpoint)
        num_classes: Number of classes (must match checkpoint)
        per_class: Images per class to generate
        names_of_classes: Optional class names
        device: Computing device
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize generator EMA
    gema = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    
    # Load EMA weights
    if 'EMA' in checkpoint:
        gema.load_state_dict(checkpoint['EMA'], strict=False)
        print("EMA weights loaded successfully")
    elif 'G' in checkpoint:
        gema.load_state_dict(checkpoint['G'], strict=False)
        print("Generator weights loaded (using regular G, not EMA)")
    else:
        raise ValueError("Checkpoint must contain 'EMA' or 'G' keys")
    
    # Generate samples
    sample_to_folder(gema, outroot, per_class=per_class, z_dim=z_dim,
                     num_classes=num_classes, names_of_classes=names_of_classes,
                     device=device)

if __name__ == '__main__':
    # Example usage
    checkpoint_path = 'checkpoints/gan_030000.pt'
    outroot = 'synthetic_images'
    
    class_names = [
        'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
        'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
        'normal',
        'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
    ]
    
    load_checkpoint_and_sample(
        checkpoint_path=checkpoint_path,
        outroot=outroot,
        z_dim=128,
        num_classes=4,
        per_class=2000,
        names_of_classes=class_names,
        device='cuda'
    )
