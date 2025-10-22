
# Create a comprehensive usage guide
usage_guide = '''# Using checkpoint.pt - Complete Guide

## Overview
Your `checkpoint.pt` file contains three saved models:
- **G**: Generator weights (creates fake images from noise)
- **D**: Discriminator weights (distinguishes real from fake)
- **EMA**: Exponential Moving Average of Generator (best quality for inference)

## File Structure
```
checkpoint.pt contains:
{
    "G": <generator state_dict>,
    "D": <discriminator state_dict>,
    "EMA": <ema shadow dict>,
    "optG": <generator optimizer state> (optional),
    "optD": <discriminator optimizer state> (optional),
    "step": <iteration number> (optional)
}
```

---

## Use Case 1: Resume Training

**When to use:** You want to continue training from where you left off.

**File:** `resume_training.py`

### Quick Start
```python
python resume_training.py
```

### Customize
```python
from resume_training import resume_train_gan
from data import build_loaders

# Load your data
train_dl, _, _, classes = build_loaders("path/to/data", batch_size=32)

# Resume training
g_ema = resume_train_gan(
    checkpoint_path="checkpoint.pt",
    train_dl=train_dl,
    num_classes=len(classes),
    z_dim=128,
    total_iters=50000,      # Train until this iteration
    start_iter=None,        # Auto-detect from checkpoint
    device="cuda",
    save_interval=5000,     # Save every 5000 iterations
    lr_g=2e-4,             # Generator learning rate
    lr_d=1e-4,             # Discriminator learning rate
)
```

### What Happens
1. ✓ Rebuilds Generator, Discriminator, EMA generator
2. ✓ Loads weights from checkpoint.pt
3. ✓ Restores optimizer states (momentum, learning rates)
4. ✓ Resumes from saved iteration number
5. ✓ Continues training with same hyperparameters
6. ✓ Saves new checkpoints periodically

---

## Use Case 2: Generate Images (Inference Only)

**When to use:** You're happy with training and just want to generate synthetic images.

**File:** `load_checkpoint_inference.py`

### Quick Start
```python
python load_checkpoint_inference.py
```

### Generate Large Dataset
```python
from load_checkpoint_inference import load_generator_from_checkpoint
from sample_gan import sample_to_folder

# Load the EMA generator
g_ema = load_generator_from_checkpoint(
    checkpoint_path="checkpoint.pt",
    num_classes=4,
    z_dim=128,
    device="cuda"
)

# Generate 1000 images per class
sample_to_folder(
    g_ema=g_ema,
    out_root="synthetic_images",
    per_class=1000,
    z_dim=128,
    num_classes=4,
    names_of_classes=["class0", "class1", "class2", "class3"],
    device="cuda"
)
```

### What Happens
1. ✓ Loads only the EMA generator (best quality)
2. ✓ Generates synthetic images organized by class
3. ✓ No training occurs (fast)

---

## Use Case 3: Manual Checkpoint Loading (Advanced)

If you need fine control, load the checkpoint manually:

```python
import torch
from gan import Generator, Discriminator

# Load checkpoint
checkpoint = torch.load("checkpoint.pt", map_location="cuda")

# Rebuild models with EXACT same architecture
G = Generator(z_dim=128, num_classes=4).to("cuda")
D = Discriminator(num_classes=4).to("cuda")
G_ema = Generator(z_dim=128, num_classes=4).to("cuda")

# Load weights
G.load_state_dict(checkpoint["G"])
D.load_state_dict(checkpoint["D"])
G_ema.load_state_dict(checkpoint["EMA"], strict=False)

# Set to eval mode for inference
G.eval()
D.eval()
G_ema.eval()

# Generate a single image
import torch
z = torch.randn(1, 128, device="cuda")  # Random noise
y = torch.tensor([0], device="cuda")    # Class label
with torch.no_grad():
    fake_img = G_ema(z, y)  # Shape: (1, 1, 256, 256)
```

---

## Understanding EMA vs G

| Model | When to Use | Quality | Purpose |
|-------|-------------|---------|---------|
| **G** | Resume training | Good | Raw trained generator |
| **EMA** | Generate images | **Best** | Smoothed, stable generator |

**Rule of thumb:**
- Use **EMA** for generating final synthetic images (inference)
- Use **G** only if you're continuing training

---

## Common Issues & Solutions

### Issue 1: "Missing keys" or "Unexpected keys"
**Cause:** Model architecture doesn't match checkpoint
**Solution:** Ensure `num_classes`, `z_dim`, and architecture are identical to training

```python
# These MUST match your training configuration
g_ema = Generator(z_dim=128, num_classes=4)  # Check these values!
```

### Issue 2: Checkpoint doesn't have optimizer states
**Cause:** Checkpoint was saved without optimizer states
**Solution:** Training will start with fresh optimizers (slightly slower convergence)

### Issue 3: CUDA out of memory
**Cause:** Batch size too large or model too big
**Solution:** Reduce batch size or use `map_location="cpu"`

```python
checkpoint = torch.load("checkpoint.pt", map_location="cpu")
```

---

## File Outputs

### Resume Training Creates:
```
checkpoints/
├── gan_000000.pt  # Checkpoint at iter 0
├── gan_005000.pt  # Checkpoint at iter 5000
└── gan_010000.pt  # Checkpoint at iter 10000

logs/
├── samples_000000.png  # Sample images at iter 0
├── samples_005000.png  # Sample images at iter 5000
└── samples_010000.png  # Sample images at iter 10000
```

### Inference Creates:
```
synthetic_images/
├── class0/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── class1/
│   └── ...
└── test_samples.png  # Quick test grid
```

---

## Quick Command Reference

```bash
# Resume training
python resume_training.py

# Generate images only
python load_checkpoint_inference.py

# Check checkpoint contents (Python)
import torch
ckpt = torch.load("checkpoint.pt", map_location="cpu")
print(ckpt.keys())  # See what's inside

# Check model architecture
from gan import Generator
g = Generator(z_dim=128, num_classes=4)
print(g)  # View layers
```

---

## Next Steps

1. **Test the checkpoint:** Run `load_checkpoint_inference.py` to verify it works
2. **Generate samples:** Create a few test images to check quality
3. **Resume or generate:**
   - If quality is good → generate full dataset
   - If quality is poor → resume training for more iterations

---

## Theory: Why EMA?

Exponential Moving Average (EMA) keeps a running average of model weights:

```
EMA_t = decay * EMA_{t-1} + (1 - decay) * Weight_t
```

**Benefits:**
- Smooths out training noise
- More stable generations
- Better FID scores (quality metric)

**Typical decay:** 0.999 (very slow averaging)

This is why EMA generator produces better images than raw G!
'''

with open("USAGE_GUIDE.md", "w") as f:
    f.write(usage_guide)

print("✓ Complete usage guide created: USAGE_GUIDE.md")
print("\n" + "="*70)
print("SUMMARY OF CREATED FILES:")
print("="*70)
print("1. resume_training.py         - Full resume training script")
print("2. load_checkpoint_inference.py - Inference-only script")
print("3. USAGE_GUIDE.md             - Complete documentation")
print("="*70)