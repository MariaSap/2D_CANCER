# GAN architecture implementation for medical image synthesis
# This module defines the Generator and Discriminator networks using ResNet-based architecture
# optimized for generating high-quality medical cancer images


import torch, torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ResBlockG(nn.Module):
    """
    Residual block for Generator with upsampling
    ResNet blocks help with gradient flow and enable training of deeper networks
    The upsampling design progressively increases spatial resolution while
    maintaining feature quality - essential for generating detailed medical images.
    """

    def __init__(self, in_ch, out_ch):
        """"
        Initialize generator residual block
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels (typically in_ch//2 for upsamlping)
        """
        super().__init__()
        # Bilinear upsampling foubles spatial dimesions (H,W) -> (2H,2W)
        # align_corners=False prevents interpolation artifacts
        self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Two 3x3 convs for feature processing
        # 3xe kernel captures local patterns interpolation artifacts
        self.c1 = nn.Conv2d(in_ch, out_ch,3,1,1) # kernel=3, stride=1, padding=1
        self.c2 = nn.Conv2d(out_ch, out_ch,3,1,1)

        # Skip connecction to preserve info flow (ResNet design)
        # 1x1 con adjusts channel dimension to match main path
        self.skip = nn.Conv2d(in_ch, out_ch, 1,1,0)

        # Batch normalization for training stability and faster convergence
        # critical for medical image generation where consisten quality is needed
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        """
        Forward pass with residual connection and upsampling

        Args:
            x: inputs feature maps
        Returns:
            torch.Tensor: upsampled feature maps with residual connection
        """
        # Main path: upsample -> conv -> norm -> activation -> conv -> norm -> activation
        y = self.ups(x)
        y = F.leaky_relu(self.bn1(self.c1(y)), 0.2, inplace=True)  # LeakyReLU prevents dead neurons
        y = F.leaky_relu(self.bn2(self.c2(y)), 0.2, inplace=True)
        # Skip path: upsample -> channel adjustment
        s = self.ups(self.skip(x))
        # residual connection: element-wise addition
        return y + s
    

class Generator(nn.Module):
    """
    Conditional Generator for synthesizing medical images

    Takes noise vector + class label and generates realistic medical images
    Uses progressive upsampling from 4x4 to 256x256 resolution through
    multiple ResNet blocks, allowing fine control over image generation
    """
    def __init__(self, z_dim=128, num_classes=4, base_ch=256, out_ch=1):
        """
        Initialize the Generator network

        Args:
            z_dim: dimension of input noise vector (latent space)
            num_classes (int): number of cancer types to generate
            base_ch (int): base number of channels (control model capacity)
            out_ch (int): output channels (1 for grayscale images)
        """
        super().__init__()
        # Class embedding: converts class labels to vectors in latent space
        # this enables conditional generation - each class gets its own embedding
        self.embed = nn.Embedding(num_classes, z_dim)

        # Initial FC layer: noise -> spatial feature maps
        # Maps from z_dim - dimensional noise to 4x4xbase_ch feature volume
        self.fc = nn.Linear(z_dim, 4*4*base_ch)


        # Progressive upsampling through ResNet blocks
        # Each block doubles spatial size while reducing channels
        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        self.b1 = ResBlockG(base_ch, base_ch//2)
        self.b2 = ResBlockG(base_ch//2, base_ch//4)
        self.b3 = ResBlockG(base_ch//4, base_ch//8)
        self.b4 = ResBlockG(base_ch//8, base_ch//16)
        self.b5 = ResBlockG(base_ch//16, base_ch//32)
        self.b6 = ResBlockG(base_ch//32, base_ch//64)

        # Final layer: convert feature maps to RGB/grayscale image
        # 1x1 conv preserves spatial dimensions while changing channels
        self.to_rgb = nn.Conv2d(base_ch//64, out_ch, 1)

    def forward(self,z,y):
        """
        Generate synthetic medical images from noise and class labels

        Args:
            z: random noise vectors [batch_size, z_dim]
            y: class labels [batch_size] (e.g., 0=benign, 1=malignant, etc)
        Returns:
            torch.tensor: Generated images in range [-1,1] with shape [batch_size, out_ch, 256, 256]
        """

        # Combine noise with class information
        # This makes generation conditioanl on the the desired cancer type
        zy = z + self.embed(y) # element-wise addition in latent space

        # Project to initial spatial feature maps (4x4)
        h = self.fc(zy).view(z.size(0), -1, 4, 4) # Reshape to [batch, channels, 4, 4]
        h = self.b1(h) # 4x4->8x8
        h = self.b2(h) # 8x8->16x16
        h = self.b3(h) # 16->32
        h = self.b4(h) # 32x32->64
        h = self.b5(h) # 64->128
        h = self.b6(h) # 128->256
        # Convert to final image with tahn activation
        # tanh outputs [-1,1 range, standard for GAN image generation]
        x = torch.tanh(self.to_rgb(h))
        return x
    
class ResBlockD(nn.Module):
    """
    Residual block for Discriminator with downsampling.
    
    Mirror of generator blocks but with downsampling instead of upsampling.
    Uses spectral normalization to prevent discriminator from becoming
    too powerful and destabilizing training.
    """
    def __init__(self, in_ch, out_ch):
        """
        Initialize discriminator residual block.
        
        Args:
            in_ch (int): Number of input channels
            out_ch (int): Number of output channels (typically in_ch * 2 for downsampling)
        """
        super().__init__()

        # 3x3 conv with spectral normalization
        # spectral norm constraints the lipschitz constant for training stability
        self.c1 = spectral_norm(nn.Conv2d(in_ch, out_ch,3,1,1))
        self.c2 = spectral_norm(nn.Conv2d(out_ch, out_ch,3,1,1))

        # Skip connection with channel adjustment
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1,1,0))

        # Average pooling for smooth downsampling (better than strided conv)
        # Reduces spatial dimensions by factor of 2
        self.pool = nn.AvgPool2d(2)
    
    def forward(self,x):
        y = F.leaky_relu(self.c1(x), 0.2, inplace=True)
        y = F.leaky_relu(self.c2(y), 0.2, inplace=True)
        y = self.pool(y)
        s = self.pool(self.skip(x))
        return y+s
    

class Discriminator(nn.Module):
    """
    Conditional Discriminator for distinguishing real from fake medical images.
    
    Uses progressive downsampling to analyze images at multiple scales,
    from fine details to global structure. The conditional design allows
    the discriminator to specialize in recognizing each cancer type.
    """
    def __init__(self, num_classes=4, in_ch=1, base_ch=64):
        """
        Initialize the Discriminator network.
        
        Args:
            num_classes (int): Number of cancer types to discriminate
            in_ch (int): Input channels (1 for grayscale medical images)
            base_ch (int): Base number of channels (controls model capacity)
        """
        super().__init__()
        # Initial layer: convert input image to feature maps
        # No spectral norm on first layer (common practice)
        self.from_rgb = spectral_norm(nn.Conv2d(in_ch, base_ch, 3,1,1))

        # Progressive downsampling through ResNet blocks
        # Each block halves spatial size while doubling channels (up to a limit)
        # 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.b1 = ResBlockD(base_ch,base_ch*2)
        self.b2 = ResBlockD(base_ch*2,base_ch*4)
        self.b3 = ResBlockD(base_ch*4,base_ch*8)
        self.b4 = ResBlockD(base_ch*8,base_ch*8)
        self.b5 = ResBlockD(base_ch*8,base_ch*8)
        self.b6 = ResBlockD(base_ch*8,base_ch*8)
        self.fc = spectral_norm(nn.Linear(base_ch*8*4*4,1))

        # Final classification layer: feature maps -> real/fake score
        # 4x4x512 -> 1 scalar output
        self.embed = spectral_norm(nn.Embedding(num_classes, base_ch*8*4*4))

    def forward(self, x, y):
        """
        Discriminate between real and fake medical images conditioned on class.
        
        Uses projection discriminator technique: combines image features with
        class information through dot product rather than concatenation.
        
        Args:
            x: Input images [batch_size, in_ch, 256, 256]
            y: Class labels [batch_size] (e.g., 0=benign, 1=malignant, etc.)
            
        Returns:
            torch.Tensor: Discriminator scores (higher = more likely real)
        """
        # Progressive feature extraction through downsampling
        h = F.leaky_relu(self.from_rgb(x), 0.2, inplace=True)  # 256x256 -> features
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        h = self.b5(h)
        h = self.b6(h) # 8x8 -> 4x4

        # Flatten spatial dimensions for classification
        h = h.view(h.size(0), -1) # [batch, channels*4*4]
        # Base discriminator output (class-agnostic)
        out = self.fc(h) # [batch, 1]


        # Projection discriminator: class-conditional term
        # Dot product between image features and class embedding
        # This allows the discriminator to specialize for each cancer type
        proj = (self.embed(y) * h).sum(dim=1, keepdim=True)  # [batch, 1]

        # Final score combines class-agnostic and class-specific components
        return out + proj