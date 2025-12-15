#!/usr/bin/env python3
"""
Example usage of VQRAE for image reconstruction and tokenization.

This script demonstrates:
1. Loading VQRAE with a pretrained configuration
2. Encoding images to discrete codes
3. Decoding from discrete codes back to images
4. Monitoring VQ loss during training
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from omegaconf import OmegaConf
from utils.model_utils import instantiate_from_config


def example_basic_usage():
    """Basic example of VQRAE forward pass."""
    print("=" * 60)
    print("Example 1: Basic VQRAE Usage")
    print("=" * 60)
    
    # Load config
    config = OmegaConf.load('configs/stage1/pretrained/VQRAE-DINOv2-B.yaml')
    
    # Instantiate model
    model = instantiate_from_config(config.stage_1)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass (reconstruction)
    with torch.no_grad():
        x_rec = model(x)
    
    print(f"Reconstruction shape: {x_rec.shape}")
    print(f"✓ Basic forward pass completed")
    print()


def example_encode_decode():
    """Example of encoding to indices and decoding."""
    print("=" * 60)
    print("Example 2: Encode to Indices and Decode")
    print("=" * 60)
    
    from stage1 import VQRAE
    
    # Create VQRAE (without pretrained weights for demo)
    model = VQRAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-base',
        encoder_params={'dinov2_path': 'facebook/dinov2-base', 'normalize': True},
        decoder_config_path='facebook/vit-mae-base',
        num_embeddings=1024,  # Smaller codebook for demo
        commitment_cost=0.25,
        noise_tau=0.0,
    )
    model.eval()
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    # Encode to discrete indices
    with torch.no_grad():
        indices = model.encode_to_indices(x)
    
    print(f"Indices shape: {indices.shape}")
    print(f"Codebook range: [{indices.min()}, {indices.max()}]")
    print(f"Number of unique codes used: {torch.unique(indices).numel()}")
    
    # Decode from indices
    with torch.no_grad():
        x_rec = model.decode_from_indices(indices)
    
    print(f"Reconstruction shape: {x_rec.shape}")
    print(f"✓ Encode/decode from indices completed")
    print()


def example_training_with_loss():
    """Example of using VQRAE in training mode with loss."""
    print("=" * 60)
    print("Example 3: Training with VQ Loss Monitoring")
    print("=" * 60)
    
    from stage1 import VQRAE
    
    # Create VQRAE
    model = VQRAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-base',
        encoder_params={'dinov2_path': 'facebook/dinov2-base', 'normalize': True},
        decoder_config_path='facebook/vit-mae-base',
        num_embeddings=512,
        commitment_cost=0.25,
        noise_tau=0.0,
    )
    model.train()
    
    # Create dummy batch
    x = torch.randn(4, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass with loss computation
    x_rec, losses = model(x, return_loss=True)
    
    print(f"Reconstruction loss: {losses['recon_loss'].item():.4f}")
    print(f"VQ loss: {losses['vq_loss'].item():.4f}")
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    
    # Simulate backward pass
    total_loss = losses['total_loss']
    total_loss.backward()
    
    print(f"✓ Training forward/backward pass completed")
    print()


def example_separate_encode_decode():
    """Example of separate encode/decode (as used in training script)."""
    print("=" * 60)
    print("Example 4: Separate Encode/Decode (Training Style)")
    print("=" * 60)
    
    from stage1 import VQRAE
    
    # Create VQRAE
    model = VQRAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-base',
        encoder_params={'dinov2_path': 'facebook/dinov2-base', 'normalize': True},
        decoder_config_path='facebook/vit-mae-base',
        num_embeddings=512,
        commitment_cost=0.25,
        noise_tau=0.0,
    )
    model.train()
    
    # Create dummy batch
    x = torch.randn(2, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    # Encode (VQ loss is computed here and backprops automatically)
    z_q = model.encode(x)
    print(f"Quantized latent shape: {z_q.shape}")
    
    # Check if VQ loss was stored
    if hasattr(model, 'last_vq_loss') and model.last_vq_loss is not None:
        print(f"VQ loss from encode: {model.last_vq_loss.item():.4f}")
    
    # Decode
    x_rec = model.decode(z_q)
    print(f"Reconstruction shape: {x_rec.shape}")
    
    # Compute reconstruction loss
    import torch.nn.functional as F
    recon_loss = F.mse_loss(x_rec, x)
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    
    # In actual training, you would do:
    # total_loss = recon_loss + lpips_loss + gan_loss
    # total_loss.backward()  # VQ loss gradients flow through automatically
    
    print(f"✓ Separate encode/decode completed")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("VQRAE Usage Examples")
    print("=" * 60 + "\n")
    
    # Note: These examples will fail without PyTorch installed
    # They are meant to demonstrate the API
    
    try:
        # Uncomment to run when PyTorch is available
        # example_basic_usage()
        # example_encode_decode()
        # example_training_with_loss()
        # example_separate_encode_decode()
        
        print("Examples ready to run (uncomment in main() when PyTorch is available)")
        print("\nKey VQRAE Features:")
        print("  - Compatible with all RAE encoders (DINOv2, SigLIP, MAE)")
        print("  - Discrete latent codes via vector quantization")
        print("  - Automatic VQ loss computation and backprop")
        print("  - Optional VQ loss monitoring via model.last_vq_loss")
        print("  - Encode to indices: model.encode_to_indices(x)")
        print("  - Decode from indices: model.decode_from_indices(indices)")
        
    except Exception as e:
        print(f"Note: Examples require PyTorch to run: {e}")
        print("See example code for API usage patterns")


if __name__ == "__main__":
    main()
