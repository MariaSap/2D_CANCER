import torch
import torch.nn.functional as F

def rand_brightness(x, strength=0.05):
    b = (torch.rand(x.size(0), 1, 1, 1, device=x.device)-0.5)*2*strength
    return x + b

def rand_contrast(x, strength=0.05):
    mean = x.mean(dim=[2,3], keepdim=True)
    c = 1.0 + (torch.rand(x.size(0),1,1,1, device=x.device)-0.5)*2*strength
    return (x-mean)*c + mean

def rand_translate(x, ratio=0.05):
    b,c,h,w = x.shape
    shift_x = (torch.randint(int(-w*ratio), int(w*ratio)+1, (b,), device=x.device)).float()
    shift_y = (torch.randint(int(-h*ratio), int(h*ratio)+1, (b,), device=x.device)).float()
    grid_x, grid_y = torch.meshgrid(torch.arange(w, device=x.device), torch.arange(h, device=x.device), indexing='xy')
    grid_x = (grid_x.unsqueeze(0)+shift_x.view(-1,1,1))/(w-1)*2-1
    grid_y = (grid_y.unsqueeze(0)+shift_y.view(-1,1,1))/(h-1)*2-1
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)

def rand_cutout(x, ratio=0.2):
    b,c,h,w = x.shape
    cut_h = int(h*ratio*torch.rand(1, device=x.device))
    cut_w = int(w*ratio*torch.rand(1, device=x.device))
    y = torch.randint(0, h-cut_w+1, (b,), device=x.device)
    x0 = torch.randint(0, w-cut_w+1, (b,), device=x.device)
    mask = torch.ones_like(x)

    for i in range(b):
        mask[i,:,y[i]:y[i]+cut_h, x0[i]:x0[i]+cut_w]=0
    return x*mask

def diffaugment(x):
    x = rand_translate(x, 0.05)
    x = rand_cutout(x, 0.15)
    x = rand_brightness(x, 0.05)
    x = rand_contrast(x, 0.05)
    return x