import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class PhysicsFeatureExtractor(nn.Module):
    """
    Physics-Aware Feature Extractor.
    Extracts gradient-based latent motion energy magnitude and directional flow orientation 
    from static images to simulate physical flow dynamics, as described in the BanglarMukh paper.
    """
    def __init__(self, eps: float = 1e-6):
        super(PhysicsFeatureExtractor, self).__init__()
        self.eps = eps
        
        # Sobel X-direction kernel
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32)
        # Sobel Y-direction kernel
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32)
        

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
        # Grayscale conversion weights: 0.2989 R + 0.5870 G + 0.1140 B
        gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
        self.register_buffer('gray_weights', gray_weights.view(1, 3, 1, 1))

    def to_grayscale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of RGB images to grayscale.
        Args:
            x: Tensor of shape (B, 3, H, W)
        Returns:
            Tensor of shape (B, 1, H, W)
        """
        if x.size(1) == 1:
            return x
        # Weighted sum across channels
        return torch.sum(x * self.gray_weights, dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extracts physics-aware motion maps from the input image.
        Args:
            x: Tensor of shape (B, 3, H, W) in range [0, 1] or normalized.
        Returns:
            physics_maps: Tensor of shape (B, 2, H, W) containing [Motion Energy, Directional Flow]
            intermediates: Dictionary of intermediate maps for visualization and analysis.
        """
        # 1. Convert to grayscale
        gray = self.to_grayscale(x)
        
        # Pad image to maintain spatial dimensions after 3x3 convolution
        padded = F.pad(gray, (1, 1, 1, 1), mode='reflect')
        
        # 2. Convolve with Sobel filters to obtain G_x and G_y gradients
        g_x = F.conv2d(padded, self.sobel_x)
        g_y = F.conv2d(padded, self.sobel_y)
        
        # 3. Calculate Latent Motion Energy Magnitude (E_motion)
        motion_energy = torch.sqrt(g_x**2 + g_y**2 + self.eps)
        
        # 4. Calculate Directional Flow Orientation (theta)
        directional_flow = torch.atan2(g_y, g_x)
        
        # Stack to create a 2-channel physics representation
        physics_maps = torch.cat([motion_energy, directional_flow], dim=1) # (B, 2, H, W)
        
        intermediates = {
            'grayscale': gray,
            'g_x': g_x,
            'g_y': g_y,
            'motion_energy': motion_energy,
            'directional_flow': directional_flow
        }
        
        return physics_maps, intermediates


class PhysicsMotionEncoder(nn.Module):
    """
    Physics Motion Encoder module.
    Processes the 2-channel physics maps (Motion Energy + Directional Flow)
    through a hierarchical convolutional architecture to produce low-rank physics embeddings.
    """
    def __init__(self, out_features: int = 256):
        super(PhysicsMotionEncoder, self).__init__()
        
        # Input channels = 2 (Motion energy, Directional flow)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.GELU()
        
        # Transition to higher semantic physics representations
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.GELU()
        
        # Adaptive pooling to handle variable image resolutions and project to fixed (8 x 8) spatial size
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Flattening 64 * 8 * 8 = 4096 
        self.flatten = nn.Flatten()
        
        # Fully connected projection to the target embedding dimension (256)
        self.fc = nn.Linear(64 * 8 * 8, out_features)
        self.ln_out = nn.LayerNorm(out_features)
        
        # Weight initialization for numerical stability in deep architectures
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, physics_maps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode physical maps into a structured latent representation.
        Args:
            physics_maps: Tensor of shape (B, 2, H, W)
        Returns:
            physics_embedding: Latent physics vector of shape (B, out_features)
        """
        # Hierarchical convolutional feature learning
        h = self.conv1(physics_maps)
        h = self.bn1(h)
        h = self.act1(h)
        
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act2(h)
        
        # Adaptive Pooling to (8, 8)
        h = self.pool(h)
        
        h = self.flatten(h)
        
        # Fully connected projection 
        out = self.fc(h)
        out = self.ln_out(out)
    
        return out
