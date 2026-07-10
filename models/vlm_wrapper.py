import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from models.physics_encoder import PhysicsFeatureExtractor, PhysicsMotionEncoder
from models.fusion import CrossConsciousFusion

class PhysicsFusedVisionTower(nn.Module):
    """
    A wrapper class that intercept's a VLM's vision tower output,
    extracts physics-aware features from the visual inputs, and fuses them
    using the CrossConsciousFusion module before passing them to the language model.
    """
    def __init__(
        self, 
        original_vision_tower: nn.Module, 
        vlm_dim: int, 
        physics_dim: int = 256, 
        num_heads: int = 8
    ):
        super(PhysicsFusedVisionTower, self).__init__()
        self.original_vision_tower = original_vision_tower
        
        # Instantiate the BanglarMukh physics and fusion modules
        self.physics_extractor = PhysicsFeatureExtractor()
        self.physics_encoder = PhysicsMotionEncoder(out_features=physics_dim)
        self.fusion = CrossConsciousFusion(
            vlm_dim=vlm_dim, 
            physics_dim=physics_dim, 
            num_heads=num_heads
        )

    def forward(self, pixel_values: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for the vision tower, incorporating physics-aware fusion.
        Args:
            pixel_values: Raw or preprocessed visual inputs of shape:
                          - Qwen2-VL: (N_patches, C, H, W) or (B, C, H, W)
                          - PaliGemma: (B, C, H, W) or (B, Seq_len, C, H, W)
        """
        #  Forward pass through original vision tower 
        visual_embeds = self.original_vision_tower(pixel_values, *args, **kwargs)
        
        img_input = pixel_values
        if len(img_input.shape) == 5:
            # Handle PaliGemma batching formats: (B, Seq_len, C, H, W) -> merge batch & seq dims
            B, S, C, H, W = img_input.shape
            img_input = img_input.view(-1, C, H, W)
        elif len(img_input.shape) == 3:

            img_input = img_input.unsqueeze(0)
            
        physics_maps, _ = self.physics_extractor(img_input)
        physics_embed = self.physics_encoder(physics_maps) 
        
        #  Perform Cross-Conscious Fusion

        if len(visual_embeds.shape) == 2:
            # Qwen2-VL 

            vis_seq = visual_embeds.unsqueeze(0)
            phys_seq = physics_embed.mean(dim=0, keepdim=True)
            fused_seq = self.fusion(vis_seq, phys_seq)
            fused_embeds = fused_seq.squeeze(0)
        else:
            if physics_embed.shape[0] != visual_embeds.shape[0]:
   
                rep_factor = visual_embeds.shape[0] // physics_embed.shape[0]
                physics_embed = physics_embed.repeat_interleave(rep_factor, dim=0)
            fused_embeds = self.fusion(visual_embeds, physics_embed)
            
        return fused_embeds


def build_banglarmukh_model(
    model_name_or_path: str,
    vlm_type: str = "qwen", 
    physics_dim: int = 256,
    num_heads: int = 8,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device_map: str = "auto"
) -> nn.Module:
    """
    Factory function to load a VLM, inject BanglarMukh's physics-aware cross-attention
    fusion into its vision tower, and apply LoRA adapters to its language modules.
    """
    from transformers import Qwen2VLForConditionalGeneration, PaliGemmaForConditionalGeneration
    
    # 1. Load the pre-trained base model
    print(f"[BanglarMukh] Loading base model: {model_name_or_path}...")
    if vlm_type.lower() == "qwen":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map=device_map
        )
        # Qwen2-VL dimension is typically model.config.hidden_size
        vlm_dim = model.config.hidden_size
        vision_module_attr = "visual"
    elif vlm_type.lower() == "gemma":
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map=device_map
        )
        vlm_dim = model.config.text_config.hidden_size
        vision_module_attr = "vision_tower"
    else:
        raise ValueError(f"Unsupported VLM type: {vlm_type}. Choose 'qwen' or 'gemma'.")
    
    # 2. Inject the custom PhysicsFusedVisionTower in-place
    print(f"[BanglarMukh] Injecting Physics-Aware Fusion (vlm_dim={vlm_dim}, physics_dim={physics_dim})...")
    original_vision = getattr(model, vision_module_attr)
    fused_vision = PhysicsFusedVisionTower(
        original_vision_tower=original_vision,
        vlm_dim=vlm_dim,
        physics_dim=physics_dim,
        num_heads=num_heads
    )
    setattr(model, vision_module_attr, fused_vision)
    
    # 3. Configure Parameter-Efficient Fine-Tuning (PEFT/LoRA)
    if use_lora:
        print("[BanglarMukh] Applying LoRA PEFT adapters to Language Model modules...")
        
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Wrap VLM model with LoRA adapters
        model = get_peft_model(model, peft_config)
        

        print("[BanglarMukh] Enabling gradients for custom physics and fusion layers...")
        for name, param in model.named_parameters():
            if any(key in name for key in ["physics_extractor", "physics_encoder", "fusion"]):
                param.requires_grad = True
                
    return model
