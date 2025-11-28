# GAN training implementation for medical image synthesis
# This module implementes the core training loop for a conditional GAN that generates medical cancer images
"""
Implements the complete GAN training pipeline with R1 regularization, exponential moving averages (EMA), 
differential augmentation, and modern training techniques for stable medical image generation.
"""

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from diffaugment import diffaugment
from gan import Generator, Discriminator
import os

def r1_penalty(d_out, x):
    """
    Compute R1 regularization penalty for the discriminator.
    
    R1 regularization helps stabilize GAN training by penalizing large gradients
    in the discriminator. This is crucial for medical image generation where
    training stability is important for consistent, high-quality outputs.
    
    Args:
        d_out: Discriminator output scores
        x: Real images (with gradients enabled)
        
    Returns:
        torch.Tensor: R1 penalty term to add to discriminator loss
    """
    # Compute gradients of discriminator output w.r.t. input images
    # create_graph=True enables higher-order gradients needed for R1 penalty
    grad = torch.autograd.grad(outputs=d_out.sum(), inputs=x, create_graph=True)[0]

    # Calculate penalty as squared L2 norm of gradients
    # This encourages the discriminator to have smoother decision boundaries
    return grad.pow(2).view(grad.size(0),-1).sum(1).mean()

class EMA:
    """
    Exponential Moving Average for model parameters.
    
    EMA maintains a running average of model weights, which typically produces
    better and more stable generations than the raw trained model. This is
    particularly important for medical image synthesis where quality is critical.
    """
    def __init__(self, model, decay=0.999):
        """
        Initialize EMA with a copy of the model's parameters.
        
        Args:
            model: The model to track (typically the Generator)
            decay: EMA decay rate (default: 0.999 for slow, stable averaging)
        """
        self.decay = decay
        self.shadow = {} # Dictionary to store EMA parameters

        # Initialize shadow parameters with current model weights
        for k, v in model.state_dict().items():
            self.shadow[k] = v.detach().clone()

    def update(self, model):
        """
        Update EMA parameters with current model weights.
        
        Formula: ema_param = decay * ema_param + (1 - decay) * current_param
        This creates a weighted average that slowly incorporates new updates.
        
        Args:
            model: Current model to update from
        """
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if torch.is_floating_point(v) and torch.is_floating_point(self.shadow[k]):
                    # Update floating-point parameters using EMA formula
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    # Keep non-floating-point buffers/params in sync by direct copy
                    # This handles integer parameters, buffers, etc.
                    self.shadow[k].copy_(v.detach())

    def copy_to(self, model):
        """
        Copy EMA parameters to a target model.
        
        This creates a stable version of the generator for inference and evaluation.
        
        Args:
            model: Target model to copy EMA weights to
        """
        with torch.no_grad():
            model.load_state_dict(self.shadow, strict=False)



def train_gan(
    train_dl, 
    num_classes=4, 
    z_dim=128, 
    iters=30000, 
    device="cuda",
    checkpoint_path=None,  # NEW: Optional checkpoint to resume from
    save_interval=5000,
    lr_g=2e-4,
    lr_d=1e-4,
    betas=(0.0, 0.99),
    ema_decay=0.999
):
    """
    Train a conditional GAN for medical image synthesis.
    
    Supports both training from scratch and resuming from a checkpoint.
    
    Args:
        train_dl: DataLoader with real medical images and class labels
        num_classes: Number of cancer types to learn
        z_dim: Dimension of noise vector
        iters: Total number of training iterations to reach
        device: Computing device
        checkpoint_path: Optional path to checkpoint file to resume from
        save_interval: Save checkpoint every N iterations
        lr_g: Generator learning rate
        lr_d: Discriminator learning rate
        betas: Adam optimizer betas
        ema_decay: EMA decay rate
    
    Returns:
        Generator: EMA version of trained generator
    """
    device = torch.device(device)
    
    # Create output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # ===== INITIALIZE MODELS =====
    print("Initializing models...")
    G = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    D = Discriminator(num_classes=num_classes).to(device)
    g_ema = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    
    # Initialize EMA
    ema = EMA(G, decay=ema_decay)
    ema.copy_to(g_ema)
    
    # ===== INITIALIZE OPTIMIZERS =====
    print("Initializing optimizers...")
    optG = optim.Adam(G.parameters(), lr=lr_g, betas=betas)
    optD = optim.Adam(D.parameters(), lr=lr_d, betas=betas)
    
    # ===== LOAD CHECKPOINT IF PROVIDED =====
    start_step = 0
    if checkpoint_path is not None:
        print(f"\n{'='*60}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        print(f"{'='*60}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model states
        G.load_state_dict(checkpoint["G"])
        D.load_state_dict(checkpoint["D"])
        
        # Restore EMA
        ema.shadow = checkpoint["EMA"]
        ema.copy_to(g_ema)
        
        # Restore optimizer states (if available)
        if "optG" in checkpoint:
            optG.load_state_dict(checkpoint["optG"])
            print("✓ Generator optimizer state restored")
        else:
            print("⚠ No generator optimizer state in checkpoint")
        
        if "optD" in checkpoint:
            optD.load_state_dict(checkpoint["optD"])
            print("✓ Discriminator optimizer state restored")
        else:
            print("⚠ No discriminator optimizer state in checkpoint")
        
        # Get starting iteration
        if "step" in checkpoint:
            start_step = checkpoint["step"]
            print(f"✓ Resuming from iteration {start_step}")
        else:
            print("⚠ No iteration info in checkpoint, starting from 0")
        
        print(f"{'='*60}\n")
    else:
        print("Training from scratch...")
    
    # ===== TRAINING LOOP =====
    print(f"Training from iteration {start_step} to {iters}...")
    print("=" * 60)
    
    step = start_step
    dl = iter(train_dl)
    print(step)
    G.train()
    D.train()


    
    while step < iters:
        # Get next batch
        try:
            real, y = next(dl)
        except StopIteration:
            dl = iter(train_dl)
            real, y = next(dl)
        
        real, y = real.to(device), y.to(device)
        
        # ===== DISCRIMINATOR TRAINING STEP =====
        # Generate fake images
        z = torch.randn(real.size(0), z_dim, device=device)
        fake = G(z, y)
        
        # Apply differential augmentation
        real_aug = diffaugment(real)
        fake_aug = diffaugment(fake.detach())
        
        # Get discriminator predictions
        real_out = D(real_aug.requires_grad_(True), y)
        fake_out = D(fake_aug, y)
        
        # Discriminator loss
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
        if step % 1000 == 0:
            print("step")
            print(f"Iter {step:6d} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f} | R1: {r1.item():.4f}")
        
        if step % save_interval == 0 and step > 0:
            print("save_interval")
            print(f"\n{'='*60}")
            print(f"Saving checkpoint at iteration {step}...")
            
            # Generate sample images
            with torch.no_grad():
                zs = torch.randn(num_classes * 8, z_dim, device=device)
                ys = torch.arange(num_classes, device=device).repeat_interleave(8)
                samples = g_ema(zs, ys)
                grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
                save_image(grid, f"logs/samples_{step:06d}.png", normalize=False)
            
            # Save checkpoint with ALL necessary state
            torch.save({
                "G": G.state_dict(),
                "D": D.state_dict(),
                "EMA": ema.shadow,
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "step": step,
                "hyperparameters": {
                    "num_classes": num_classes,
                    "z_dim": z_dim,
                    "lr_g": lr_g,
                    "lr_d": lr_d,
                    "betas": betas,
                    "ema_decay": ema_decay
                }
            }, f"checkpoints/gan_{step:06d}.pt")
            
            print(f"✓ Checkpoint saved: checkpoints/gan_{step:06d}.pt")
            print(f"✓ Samples saved: logs/samples_{step:06d}.png")
            print(f"{'='*60}\n")
        
        step += 1
    
    print("\n" + "=" * 60)
    print(f"Training completed! Final iteration: {step}")
    print("=" * 60)
    
    return g_ema


















# # def train_gan(train_dl, num_classes=4, z_dim=128, iters=30000, device="cuda"):
#     """
#     Train a conditional GAN for medical image synthesis.
    
#     This implements a modern GAN training procedure optimized for generating
#     high-quality medical images. The conditional setup allows generating
#     specific types of cancer images on demand.
    
#     Args:
#         train_dl: DataLoader with real medical images and class labels
#         num_classes: Number of cancer types to learn (default: 4)
#         z_dim: Dimension of noise vector (latent space size)
#         iters: Number of training iterations
#         device: Computing device ('cuda' recommended for speed)
        
#     Returns:
#         Generator: EMA version of trained generator for stable inference
#     """

#     # Initialize networks
#     # Generator: Creates fake images from noise + class to)
#     G = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
#     # Discriminator: Distinguishes real from fake images (class-conditional)
#     D = Discriminator(num_classes=num_classes).to(device)
#     # EMA generator for stable, high-quality inference
#     g_ema = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
#     ema = EMA(G, 0.999)  # Track EMA of generator weights
#     ema.copy_to(g_ema)   # Initialize EMA generator
    
   
#     # Optimizers with different learning rates (common GAN practice)
#     # Generator uses faster learning rate to prevent overpowering discriminator
#     optG = optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.99))
#     # Discriminator uses slower learning rate to stay competitive
#     optD = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.99))
    
#     step = 0
#     dl = iter(train_dl)  # Create iterator for cycling through data

#     while step<iters:
#         # Get next batch of real images and labels
#         try:
#             real, y = next(dl)
#         except StopIteration:
#             # Restart iterator when dataset is exhausted
#             dl = iter(train_dl)
#             real, y = next(dl)
#         real, y = real.to(device), y.to(device)
#         assert real.is_cuda and y.is_cuda, "Batch not on GPU"

#         # === DISCRIMINATOR TRAINING STEP ===
#         D.train(); G.train()

#         # Generate fake images for this batch
#         z = torch.randn(real.size(0), z_dim, device=device) # Sample noise
#         fake = G(z,y)  # Generate fake images with same class labels

#         # Apply differential augmentation to both real and fake images
#         # This helps prevent discriminator overfitting and improves training stability
#         real_aug = diffaugment(real) 
#         fake_aug= diffaugment(fake.detach()) # Detach to avoid generator gradients

#         # Get discriminator predictions
#         real_out = D(real_aug.requires_grad_(True),y)  # Enable gradients for R1 penalty
#         fake_out = D(fake_aug, y)

#         # Discriminator loss: maximize log(D(real)) + log(1 - D(fake))
#         # Using softplus for numerical stability: -log(sigmoid(x)) = softplus(-x)
#         d_loss = (F.softplus(fake_out).mean() + F.softplus(-real_out).mean())

#         # Add R1 regularization to prevent discriminator gradients from exploding
#         r1 = r1_penalty(real_out, real_aug) * 10.0
#         # Update discriminator
#         (d_loss +  r1).backward()
#         optD.step()  # Apply gradient updates
#         optD.zero_grad()  # Clear gradients
#         G.zero_grad()     # Clear generator gradients (in case of shared computation)

#         # === GENERATOR TRAINING STEP ===
#         # Generate new fake images for generator training
#         z = torch.randn(real.size(0), z_dim, device=device)
#         y2 = y # Use same class labels as real images
#         fake = G(z, y2)

#         # Get discriminator's opinion on generated images
#         fake_out = D(diffaugment(fake), y2)

#         # Generator loss: maximize log(D(fake)) to fool discriminator
#         # Equivalent to minimizing -log(D(fake)) = softplus(-fake_out)
#         g_loss = F.softplus(-fake_out).mean()
#         # Update generator
#         g_loss.backward()
#         optG.step()      # Apply gradient updates
#         optG.zero_grad() # Clear generator gradients
#         D.zero_grad()    # Clear discriminator gradients
        
#         # Update EMA generator with new weights
#         ema.update(G); ema.copy_to(g_ema)


#         # === LOGGING AND CHECKPOINTING ===
#         if step % 1000 == 0:
#             # Generate sample images for visual monitoring
#             with torch.no_grad():
#                 # Create a grid showing samples from each class
#                 zs = torch.randn(num_classes*8, z_dim, device=device)
#                 ys = torch.arange(num_classes, device=device).repeat_interleave(8)
#                 samples = g_ema(zs, ys)
#                 # Arrange samples in a grid (8 samples per class)
#                 grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1,1))
#                 save_image(grid, f"logs/samples_{step:06d}.png", normalize=False)
#             # Save model checkpoints for resuming training or inference
#             torch.save({"G": G.state_dict(), # Generator weights
#                         "D": D.state_dict(), # Discriminator weights 
#                         "EMA": ema.shadow,   # EMA generator weights,
#                         "optG": optG.state_dict(), # Generator optimizer
#                         "optD": optD.state_dict(), # Discriminator optimizer
#                         "step": step               # Iteration counter
#                         }, f"checkpoints/gan_{step:06d}.pt")
#         step += 1

#     # Return the EMA generator for stable, high-quality image generation
#     return g_ema