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


def train_gan(train_dl, num_classes=4, z_dim=128, iters=30000, device="cuda"):
    """
    Train a conditional GAN for medical image synthesis.
    
    This implements a modern GAN training procedure optimized for generating
    high-quality medical images. The conditional setup allows generating
    specific types of cancer images on demand.
    
    Args:
        train_dl: DataLoader with real medical images and class labels
        num_classes: Number of cancer types to learn (default: 4)
        z_dim: Dimension of noise vector (latent space size)
        iters: Number of training iterations
        device: Computing device ('cuda' recommended for speed)
        
    Returns:
        Generator: EMA version of trained generator for stable inference
    """

    # Initialize networks
    # Generator: Creates fake images from noise + class to)
    G = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    # Discriminator: Distinguishes real from fake images (class-conditional)
    D = Discriminator(num_classes=num_classes).to(device)
    # EMA generator for stable, high-quality inference
    g_ema = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    ema = EMA(G, 0.999)  # Track EMA of generator weights
    ema.copy_to(g_ema)   # Initialize EMA generator
    
   
    # Optimizers with different learning rates (common GAN practice)
    # Generator uses faster learning rate to prevent overpowering discriminator
    optG = optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.99))
    # Discriminator uses slower learning rate to stay competitive
    optD = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.99))
    
    step = 0
    dl = iter(train_dl)  # Create iterator for cycling through data

    while step<iters:
        # Get next batch of real images and labels
        try:
            real, y = next(dl)
        except StopIteration:
            # Restart iterator when dataset is exhausted
            dl = iter(train_dl)
            real, y = next(dl)
        real, y = real.to(device), y.to(device)
        assert real.is_cuda and y.is_cuda, "Batch not on GPU"

        # === DISCRIMINATOR TRAINING STEP ===
        D.train(); G.train()

        # Generate fake images for this batch
        z = torch.randn(real.size(0), z_dim, device=device) # Sample noise
        fake = G(z,y)  # Generate fake images with same class labels

        # Apply differential augmentation to both real and fake images
        # This helps prevent discriminator overfitting and improves training stability
        real_aug = diffaugment(real) 
        fake_aug= diffaugment(fake.detach()) # Detach to avoid generator gradients

        # Get discriminator predictions
        real_out = D(real_aug.requires_grad_(True),y)  # Enable gradients for R1 penalty
        fake_out = D(fake_aug, y)

        # Discriminator loss: maximize log(D(real)) + log(1 - D(fake))
        # Using softplus for numerical stability: -log(sigmoid(x)) = softplus(-x)
        d_loss = (F.softplus(fake_out).mean() + F.softplus(-real_out).mean())

        # Add R1 regularization to prevent discriminator gradients from exploding
        r1 = r1_penalty(real_out, real_aug) * 10.0
        # Update discriminator
        (d_loss +  r1).backward()
        optD.step()  # Apply gradient updates
        optD.zero_grad()  # Clear gradients
        G.zero_grad()     # Clear generator gradients (in case of shared computation)

        # === GENERATOR TRAINING STEP ===
        # Generate new fake images for generator training
        z = torch.randn(real.size(0), z_dim, device=device)
        y2 = y # Use same class labels as real images
        fake = G(z, y2)

        # Get discriminator's opinion on generated images
        fake_out = D(diffaugment(fake), y2)

        # Generator loss: maximize log(D(fake)) to fool discriminator
        # Equivalent to minimizing -log(D(fake)) = softplus(-fake_out)
        g_loss = F.softplus(-fake_out).mean()
        # Update generator
        g_loss.backward()
        optG.step()      # Apply gradient updates
        optG.zero_grad() # Clear generator gradients
        D.zero_grad()    # Clear discriminator gradients
        
        # Update EMA generator with new weights
        ema.update(G); ema.copy_to(g_ema)


        # === LOGGING AND CHECKPOINTING ===
        if step % 5000 == 0:
            # Generate sample images for visual monitoring
            with torch.no_grad():
                # Create a grid showing samples from each class
                zs = torch.randn(num_classes*8, z_dim, device=device)
                ys = torch.arange(num_classes, device=device).repeat_interleave(8)
                samples = g_ema(zs, ys)
                # Arrange samples in a grid (8 samples per class)
                grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1,1))
                save_image(grid, f"logs/samples_{step:06d}.png", normalize=False)
            # Save model checkpoints for resuming training or inference
            torch.save({"G": G.state_dict(), # Generator weights
                        "D": D.state_dict(), # Discriminator weights 
                        "EMA": ema.shadow,   # EMA generator weights,
                        "optG": optG.state_dict(), # Generator optimizer (NEW!)
                        "optD": optD.state_dict(), # Discriminator optimizer (NEW!)
                        "step": step               # Iteration counter (NEW!)
                        }, f"checkpoints/gan_{step:06d}.pt")
        step += 1

    # Return the EMA generator for stable, high-quality image generation
    return g_ema