# Create the complete resume training script - fixing indentation


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from pathlib import Path

# Import your existing modules
from data import build_loaders
from train_gan import r1_penalty, EMA
from gan import Generator, Discriminator
from diffaugment import diffaugment


def resume_train_gan(
    checkpoint_path,
    train_dl,
    num_classes=4,
    z_dim=128,
    total_iters=30000,
    start_iter=None,
    device="cuda",
    save_interval=5000,
    lr_g=2e-4,
    lr_d=1e-4,
    betas=(0.0, 0.99),
    ema_decay=0.999,
):
    """
    Resume training a conditional GAN from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint.pt file
        train_dl: DataLoader with real medical images and class labels
        num_classes (int): Number of cancer types (must match checkpoint)
        z_dim (int): Latent dimension (must match checkpoint)
        total_iters (int): Total number of training iterations to reach
        start_iter (int): Override the iteration counter (if None, read from checkpoint)
        device (str): Computing device ('cuda' or 'cpu')
        save_interval (int): Save checkpoint every N iterations
        lr_g (float): Generator learning rate
        lr_d (float): Discriminator learning rate
        betas (tuple): Adam optimizer betas
        ema_decay (float): EMA decay rate
        
    Returns:
        Generator: The trained EMA generator
    """
    device = torch.device(device)
    
    # Create output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"Resuming training from: {checkpoint_path}")
    
    # ===== STEP 1: REBUILD MODELS WITH SAME ARCHITECTURE =====
    print("Step 1: Rebuilding model architectures...")
    G = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    D = Discriminator(num_classes=num_classes).to(device)
    g_ema = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    
    # ===== STEP 2: REBUILD OPTIMIZERS =====
    print("Step 2: Rebuilding optimizers...")
    optG = optim.Adam(G.parameters(), lr=lr_g, betas=betas)
    optD = optim.Adam(D.parameters(), lr=lr_d, betas=betas)
    
    # ===== STEP 3: LOAD CHECKPOINT =====
    print("Step 3: Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    G.load_state_dict(checkpoint["G"])
    D.load_state_dict(checkpoint["D"])

    
    # ===== STEP 4: RESTORE EMA =====
    print("Step 4: Restoring EMA generator...")
    # Rebuild EMA object and restore shadow weights
    ema = EMA(G, decay=ema_decay)
    ema.shadow = checkpoint["EMA"]  # Restore the EMA shadow dict
    ema.copy_to(g_ema)  # Copy EMA weights to g_ema model
    
    # ===== STEP 5: RESTORE OPTIMIZER STATES (if available) =====
    print("Step 5: Restoring optimizer states...")
    if "optG" in checkpoint:
        optG.load_state_dict(checkpoint["optG"])
        print("  ✓ Generator optimizer state restored")
    else:
        print("  ⚠ No generator optimizer state in checkpoint (starting fresh)")
        
    if "optD" in checkpoint:
        optD.load_state_dict(checkpoint["optD"])
        print(" ✓ Discriminator optimizer state restored")
    else:
        print(" ⚠ No discriminator optimizer state in checkpoint (starting fresh)")
    
    # ===== STEP 6: DETERMINE STARTING ITERATION =====
    if start_iter is not None:
        step = start_iter
        print(f"Step 6: Overriding start iteration to {step}")
    elif "step" in checkpoint:
        step = checkpoint["step"]
        print(f"Step 6: Resuming from iteration {step}")
    else:
        step = 0
        print(f"Step 6: No iteration info in checkpoint, starting from {step}")
    
    # ===== STEP 7: RESUME TRAINING LOOP =====
    print(f"\\nResuming training from iteration {step} to {total_iters}...")
    print("=" * 60)
    
    dl = iter(train_dl)
    G.train()
    D.train()
    
    while step < total_iters:

        # Get next batch
        try:
            real, y = next(dl)
        except StopIteration:
            dl = iter(train_dl)
            real, y = next(dl)
        
        real, y = real.to(device), y.to(device)
        
        # ===== DISCRIMINATOR TRAINING STEP =====
        D.train()
        G.train()
        
        # Generate fake images
        z = torch.randn(real.size(0), z_dim, device=device)
        fake = G(z, y)
        
        # Apply differential augmentation
        real_aug = diffaugment(real)
        fake_aug = diffaugment(fake.detach())
        
        # Get discriminator predictions
        real_out = D(real_aug.requires_grad_(True), y)
        fake_out = D(fake_aug, y)
        
        # Discriminator loss with softplus
        d_loss = F.softplus(fake_out).mean() + F.softplus(-real_out).mean()
        
        # R1 regularization
        r1 = r1_penalty(real_out, real_aug) * 10.0
        
        # Update discriminator
        (d_loss + r1).backward()
        optD.step()
        optD.zero_grad()
        G.zero_grad()
        
        # ===== GENERATOR TRAINING STEP =====
        z = torch.randn(real.size(0), z_dim, device=device)
        fake = G(z, y)
        fake_out = D(diffaugment(fake), y)
        
        # Generator loss
        g_loss = F.softplus(-fake_out).mean()
        
        # Update generator
        g_loss.backward()
        optG.step()
        optG.zero_grad()
        D.zero_grad()
        
        # Update EMA
        ema.update(G)
        ema.copy_to(g_ema)
        
        # ===== LOGGING AND CHECKPOINTING =====
        if step % 3000 == 0:
            print(f"Iter {step:6d} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f} | R1: {r1.item():.4f}")
        
        if step % save_interval == 0:
            print(f"\\n{'='*60}")
            print(f"Saving checkpoint at iteration {step}...")
            
            # Generate sample images
            with torch.no_grad():
                zs = torch.randn(num_classes * 8, z_dim, device=device)
                ys = torch.arange(num_classes, device=device).repeat_interleave(8)
                samples = g_ema(zs, ys)
                grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
                save_image(grid, f"logs/samples_{step:06d}.png", normalize=False)
            
            # Save checkpoint with optimizer states and iteration counter
            torch.save({
                "G": G.state_dict(),
                "D": D.state_dict(),
                "EMA": ema.shadow,
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "step": step,
            }, f"checkpoints/Resumegan_{step:06d}.pt")
            
            print(f"✓ Checkpoint saved: checkpoints/Resumegan_{step:06d}.pt")
            print(f"✓ Samples saved: logs/samples_{step:06d}.png")
            print(f"{'='*60}\\n")
        
        step += 1
    
    print("\\n" + "=" * 60)
    print(f"Training completed! Final iteration: {step}")
    print("=" * 60)
    
    return g_ema


if __name__ == "__main__":
    # Example usage
    
    # Configuration
    DATA_ROOT = Path(r"C:data")  # Adjust this to your data path
    CHECKPOINT_PATH = Path(r"C:\Users\sapounaki.m\Desktop\2D_CANCER\checkpoints\Resumegan_000100.pt")   # Path to your checkpoint file
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file '{CHECKPOINT_PATH}' not found!")
        print("Please provide the correct path to your checkpoint.pt file.")
        exit(1)
    
    # Load data
    print("Loading data...")
    train_dl, val_dl, test_dl, classes = build_loaders(DATA_ROOT, batch_size=32)
    print(f"Dataset loaded: {len(classes)} classes")
    
    # Resume training
    g_ema = resume_train_gan(
        checkpoint_path=CHECKPOINT_PATH,
        train_dl=train_dl,
        num_classes=len(classes),
        z_dim=128,
        total_iters=15000,  # Continue training until this many iterations
        start_iter=None,  # Will be read from checkpoint
        device="cuda",
        save_interval=100,
        lr_g=2e-4,
        lr_d=1e-4,
        betas=(0.0, 0.99),
        ema_decay=0.999,
    )
    
    print("\\n✓ Training resumed and completed successfully!")
    print("\\nTo use the trained generator for inference:")
    print("  from sample_gan import sample_to_folder")
    print("  sample_to_folder(g_ema, 'output_dir', per_class=100, num_classes=len(classes))")




# print("✓ Complete resume training script created: resume_training.py")
# print("\n" + "="*70)
# print("KEY FEATURES OF THE SCRIPT:")
# print("="*70)
# print("✓ Rebuilds Generator, Discriminator, and EMA generator")
# print("✓ Loads weights from checkpoint.pt (G, D, EMA)")
# print("✓ Restores optimizer states if available")
# print("✓ Handles iteration counter (reads from checkpoint or starts fresh)")
# print("✓ Full training loop with:")
# print("  - R1 gradient penalty")
# print("  - Differential augmentation")  
# print("  - EMA updates")
# print("  - Softplus loss (non-saturating)")
# print("✓ Periodic checkpointing with ALL states saved")
# print("✓ Sample image generation for monitoring")
# print("="*70)