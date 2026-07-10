import torch
import torch.nn as nn
from typing import Optional

class CrossConsciousFusion(nn.Module):
    """
    Cross-Conscious Fusion Module.
    Fuses visual token embeddings with low-rank physics embeddings using a multi-head
    cross-attention mechanism. The visual embeddings act as queries, while the projected
    physics embedding acts as keys and values.
    """
    def __init__(self, vlm_dim: int, physics_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super(CrossConsciousFusion, self).__init__()
        
        self.vlm_dim = vlm_dim
        self.physics_dim = physics_dim
        self.num_heads = num_heads
        
        # Linear projection to map physics embedding from physics_dim to vlm_dim
        self.physics_proj = nn.Linear(physics_dim, vlm_dim)
    
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vlm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Normalization and dropout layers for regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(vlm_dim)
        
        # Optional feed-forward layer post-attention to further contextualize fused embeddings
        self.ffn = nn.Sequential(
            nn.Linear(vlm_dim, 4 * vlm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * vlm_dim, vlm_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(vlm_dim)
        
        self._init_weights()

    def _init_weights(self):
        # Professional weight initialization
        nn.init.xavier_normal_(self.physics_proj.weight)
        if self.physics_proj.bias is not None:
            nn.init.constant_(self.physics_proj.bias, 0.0)

    def forward(self, visual_embeds: torch.Tensor, physics_embed: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs cross-conscious fusion of visual and physics embeddings.
        Args:
            visual_embeds: Tensor of shape (B, N_visual, D_vlm) representing visual tokens.
            physics_embed: Tensor of shape (B, D_physics) representing global/local physical features.
            attn_mask: Optional attention mask for the cross-attention block.
        Returns:
            fused_embeds: Fused representations of shape (B, N_visual, D_vlm).
        """
        # 1. Project physical features to VLM embedding space
        physics_projected = self.physics_proj(physics_embed)
        
        # 2. Expand dimensions to represent a single physical sequence token
        physics_token = physics_projected.unsqueeze(1)
        
        # 3. Cross-attention: Visual tokens query the physical token
        attn_out, _ = self.cross_attn(
            query=visual_embeds,
            key=physics_token,
            value=physics_token,
            key_padding_mask=None,
            attn_mask=attn_mask
        )
        
        # 4. Residual Connection and Layer Normalization
        fused = self.norm(visual_embeds + self.attn_dropout(attn_out))
        
        # 5. Feed-forward refinement network
        ffn_out = self.ffn(fused)
        fused_embeds = self.ffn_norm(fused + ffn_out)
        
        return fused_embeds
