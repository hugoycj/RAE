from transformers import DINOv3ViTModel
from torch import nn
import torch
from . import register_encoder


@register_encoder()
class Dinov3withNorm(nn.Module):
    """DINOv3 Vision Transformer encoder with normalized output.

    Supports DINOv3 ViT models like facebook/dinov3-vitl16-pretrain-lvd1689m.
    Uses the official DINOv3ViTModel from transformers (requires v4.56.0+).

    DINOv3 outputs: 1 CLS token + 4 register tokens + N patch tokens
    For 224x224 with patch_size=16: 1 + 4 + 196 = 201 tokens
    """

    def __init__(
        self,
        dinov3_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = DINOv3ViTModel.from_pretrained(dinov3_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = DINOv3ViTModel.from_pretrained(dinov3_path, local_files_only=False)

        self.encoder.requires_grad_(False)

        # Remove affine parameters from final layernorm for normalized output
        if normalize:
            # DINOv3ViTModel uses layernorm attribute
            if hasattr(self.encoder, 'layernorm') and self.encoder.layernorm is not None:
                self.encoder.layernorm.elementwise_affine = False
                self.encoder.layernorm.weight = None
                self.encoder.layernorm.bias = None

        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size

        # DINOv3 uses 4 register tokens by default
        self.num_register_tokens = getattr(self.encoder.config, 'num_register_tokens', 4)
        # Skip CLS token (1) + register tokens (default 4)
        self.unused_token_num = 1 + self.num_register_tokens

    def dinov3_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting only patch features.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Patch features of shape (B, N, hidden_size) where N = (H/patch_size) * (W/patch_size)
        """
        outputs = self.encoder(x, output_hidden_states=True)
        # Remove CLS and register tokens, keep only patch embeddings
        # Output shape: (B, 1 + num_register_tokens + num_patches, hidden_size)
        # We want: (B, num_patches, hidden_size)
        image_features = outputs.last_hidden_state[:, self.unused_token_num:]
        return image_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov3_forward(x)
