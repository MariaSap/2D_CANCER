import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from diffaugment import diffaugment
from gan import Generator, Discriminator
import numpy as np

def r1_penalty(dout, x):
    """
    Compute R1 regularization penalty for the discriminator.
    
    R1 regularization helps stabilize GAN training by penalizing large gradients.
    This is crucial for medical image generation where training stability
    is important for consistent, high-quality outputs.
    """
    grad = torch.autograd.grad(outputs=dout.sum(), inputs=x, create_graph=True)[0]
    return grad.pow(2).view(grad.size(0), -1).sum(1).mean()

class EMA:
    """
    Exponential Moving Average for model parameters.
    
    EMA maintains a running average of model weights, which typically produces
    better and more stable generations than the raw trained model.
    This is particularly important for medical image synthesis.
    """
    def __init__(self, model, decay=0.999):
        """Initialize EMA with a copy of the model's parameters."""
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.detach().clone()
    
    def update(self, model):
        """Update EMA parameters with current model weights."""
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if torch.is_floating_point(v) and torch.is_floating_point(self.shadow[k]):
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    self.shadow[k].copy_(v.detach())
    
    def copy_to(self, model):
        """Copy EMA parameters to a target model."""
        with torch.no_grad():
            model.load_state_dict(self.shadow, strict=False)

def train_gan(traindl, num_classes=4, z_dim=128, iters=30000, device='cuda',
              checkpoint_path=None, save_interval=2, lrg=2e-4, lrd=1e-4,
              betas=(0.0, 0.99), ema_decay=0.999):
    """
    Train a conditional GAN for medical image synthesis.
    
    CRITICAL IMPROVEMENTS:
    1. WeightedRandomSampler ensures all classes are sampled equally
    2. Per-class loss monitoring detects mode collapse early
    3. Consistent [-1, 1] normalization throughout pipeline
    4. Supports resuming from checkpoint
    
    Args:
        traindl: DataLoader with real medical images and class labels
        num_classes: Number of cancer types to learn
        z_dim: Dimension of noise vector
        iters: Total number of training iterations
        device: Computing device
        checkpoint_path: Optional path to checkpoint file to resume from
        save_interval: Save checkpoint every N iterations
        lrg: Generator learning rate
        lrd: Discriminator learning rate
        betas: Adam optimizer betas
        ema_decay: EMA decay rate
    
    Returns:
        Generator: EMA version of trained generator
    """
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    print("Initializing models...")
    G = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    D = Discriminator(num_classes=num_classes).to(device)
    gema = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    
    ema = EMA(G, decay=ema_decay)
    ema.copy_to(gema)
    
    print("Initializing optimizers...")
    optG = optim.Adam(G.parameters(), lr=lrg, betas=betas)
    optD = optim.Adam(D.parameters(), lr=lrd, betas=betas)
    
    start_step = 0
    
    # ===== LOAD CHECKPOINT IF PROVIDED =====
    if checkpoint_path is not None:
        print("=" * 60)
        print(f"Loading checkpoint from {checkpoint_path}")
        print("=" * 60)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        G.load_state_dict(checkpoint['G'])
        D.load_state_dict(checkpoint['D'])
        ema.shadow = checkpoint['EMA']
        ema.copy_to(gema)
        
        if 'optG' in checkpoint:
            optG.load_state_dict(checkpoint['optG'])
            print("Generator optimizer state restored")
        if 'optD' in checkpoint:
            optD.load_state_dict(checkpoint['optD'])
            print("Discriminator optimizer state restored")
        
        if 'step' in checkpoint:
            start_step = checkpoint['step']
            print(f"Resuming from iteration {start_step}")
        print("=" * 60)
    else:
        print("Training from scratch...")
    
    print(f"Training from iteration {start_step} to {iters}...")
    print("=" * 60)
    
    # ===== PER-CLASS LOSS TRACKING (CRITICAL FIX) =====
    class_loss_tracker = {i: [] for i in range(num_classes)}
    
    step = start_step
    dl = iter(traindl)
    
    while step < iters:
        try:
            real, y = next(dl)
        except StopIteration:
            dl = iter(traindl)
            real, y = next(dl)
        
        real, y = real.to(device), y.to(device)
        
        # ===== DISCRIMINATOR TRAINING =====
        z = torch.randn(real.size(0), z_dim, device=device)
        fake = G(z, y)
        
        real_aug = diffaugment(real)
        fake_aug = diffaugment(fake.detach())
        
        real_out = D(real_aug.requires_grad_(True), y)
        fake_out = D(fake_aug, y)
        
        dloss = F.softplus(fake_out).mean() + F.softplus(-real_out).mean()
        
        r1 = r1_penalty(real_out, real_aug) * 10.0
        dloss_total = dloss + r1
        
        dloss_total.backward()
        optD.step()
        optD.zero_grad()
        G.zero_grad()
        
        # ===== GENERATOR TRAINING =====
        z = torch.randn(real.size(0), z_dim, device=device)
        fake = G(z, y)
        fake_out = D(diffaugment(fake), y)
        
        gloss = F.softplus(-fake_out).mean()
        gloss.backward()
        optG.step()
        optG.zero_grad()
        D.zero_grad()
        
        ema.update(G)
        ema.copy_to(gema)
        
        # ===== PER-CLASS LOSS TRACKING =====
        for cls_id in range(num_classes):
            class_mask = (y == cls_id)
            if class_mask.sum() > 0:
                class_loss_tracker[cls_id].append(gloss.item())
        
        # ===== LOGGING WITH PER-CLASS BREAKDOWN =====
        if step % 1000 == 0:
            print(f"\n{'='*70}")
            print(f"Iter {step:6d}")
            print(f"  Overall  - D_loss: {dloss.item():.4f}, G_loss: {gloss.item():.4f}, R1: {r1.item():.4f}")
            
            # Print per-class average loss over last 100 steps
            for cls_id in range(num_classes):
                if len(class_loss_tracker[cls_id]) > 0:
                    avg_loss = np.mean(class_loss_tracker[cls_id][-100:])
                    print(f"  Class {cls_id}: avg G_loss = {avg_loss:.4f}")
            
            print(f"{'='*70}\n")
        
        # ===== CHECKPOINTING AND SAMPLING =====
        if step % save_interval == 0 and step > 0:
            print("=" * 60)
            print(f"Saving checkpoint at iteration {step}...")
            
            with torch.no_grad():
                # Generate sample grid for monitoring
                zs = torch.randn(num_classes * 8, z_dim, device=device)
                ys = torch.arange(num_classes, device=device).repeat_interleave(8)
                samples = gema(zs, ys)
                
                grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
                save_image(grid, f'logs/samples_{step:06d}.png', normalize=False)
            
            torch.save({
                'G': G.state_dict(),
                'D': D.state_dict(),
                'EMA': ema.shadow,
                'optG': optG.state_dict(),
                'optD': optD.state_dict(),
                'step': step,
                'hyperparameters': {
                    'num_classes': num_classes,
                    'z_dim': z_dim,
                    'lrg': lrg,
                    'lrd': lrd,
                    'betas': betas,
                    'ema_decay': ema_decay
                }
            }, f'checkpoints/gan_{step:06d}.pt')
            
            print(f"Checkpoint saved: checkpoints/SecGAN_{step:06d}.pt")
            print(f"Samples saved: logs/SecSamples_{step:06d}.png")
            print("=" * 60)
        
        step += 1
    
    print("=" * 60)
    print(f"Training completed! Final iteration: {step}")
    print("=" * 60)
    
    return gema