# src/sample_gan.py
import os, torch
from torchvision.utils import save_image
from gan import Generator

def sample_to_folder(g_ema, out_root, per_class=2000, z_dim=128, num_classes=4, device="cuda"):
    os.makedirs(out_root, exist_ok=True)
    for c in range(num_classes):
        cls_dir = os.path.join(out_root, str(c))
        os.makedirs(cls_dir, exist_ok=True)
        n = 0 
        while n < per_class:
            bs = min(64, per_class - n)
            z = torch.randn(bs, z_dim, device=device)
            y = torch.full((bs,), c, device=device, dtype=torch.long)
            with torch.no_grad():
                x = g_ema(z, y)
            x = (x.clamp(-1,1)+1)/2.0
            for i in range(bs):
                save_image(x[i], os.path.join(cls_dir, f"{n+i:06d}.png"))
            n += bs
