#!/usr/bin/env python3
"""
Simple test script to verify VQRAE implementation.
This script tests the basic functionality without requiring full model weights.
"""

import torch
import sys
import os

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
sys.path.insert(0, src_dir)

from stage1 import VQRAE


def test_vqrae_initialization():
    """Test that VQRAE can be initialized."""
    print("Test 1: VQRAE Initialization")
    try:
        # Create minimal VQRAE without pretrained weights
        model = VQRAE(
            encoder_cls='Dinov2withNorm',
            encoder_config_path='facebook/dinov2-base',
            encoder_input_size=224,
            encoder_params={'dinov2_path': 'facebook/dinov2-base', 'normalize': True},
            decoder_config_path='facebook/vit-mae-base',
            decoder_patch_size=16,
            pretrained_decoder_path=None,
            num_embeddings=512,  # Small codebook for testing
            commitment_cost=0.25,
            vq_decay=0.99,
            noise_tau=0.0,
            reshape_to_2d=True,
            normalization_stat_path=None,
        )
        print("✓ VQRAE initialized successfully")
        print(f"  - Latent dim: {model.latent_dim}")
        print(f"  - Codebook size: {model.num_embeddings}")
        print(f"  - Quantizer embedding shape: {model.quantizer.embedding.shape}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize VQRAE: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vqrae_forward():
    """Test forward pass through VQRAE."""
    print("\nTest 2: VQRAE Forward Pass")
    try:
        model = VQRAE(
            encoder_cls='Dinov2withNorm',
            encoder_config_path='facebook/dinov2-base',
            encoder_input_size=224,
            encoder_params={'dinov2_path': 'facebook/dinov2-base', 'normalize': True},
            decoder_config_path='facebook/vit-mae-base',
            decoder_patch_size=16,
            pretrained_decoder_path=None,
            num_embeddings=512,
            commitment_cost=0.25,
            vq_decay=0.99,
            noise_tau=0.0,
            reshape_to_2d=True,
            normalization_stat_path=None,
        )
        model.eval()
        
        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Test basic forward
        print("  Testing basic forward pass...")
        with torch.no_grad():
            x_rec = model(x)
        print(f"  ✓ Input shape: {x.shape}")
        print(f"  ✓ Output shape: {x_rec.shape}")
        assert x.shape == x_rec.shape, "Output shape should match input shape"
        
        # Test forward with loss
        print("  Testing forward with loss...")
        model.train()
        x_rec, losses = model(x, return_loss=True)
        print(f"  ✓ Reconstruction loss: {losses['recon_loss'].item():.4f}")
        print(f"  ✓ VQ loss: {losses['vq_loss'].item():.4f}")
        print(f"  ✓ Total loss: {losses['total_loss'].item():.4f}")
        
        # Test forward with indices
        print("  Testing forward with indices...")
        model.eval()
        with torch.no_grad():
            x_rec, indices = model(x, return_indices=True)
        print(f"  ✓ Indices shape: {indices.shape}")
        print(f"  ✓ Indices range: [{indices.min()}, {indices.max()}]")
        assert indices.min() >= 0 and indices.max() < 512, "Indices should be in valid range"
        
        return True
    except Exception as e:
        print(f"✗ Failed forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vqrae_encode_decode():
    """Test encode and decode separately."""
    print("\nTest 3: VQRAE Encode/Decode")
    try:
        model = VQRAE(
            encoder_cls='Dinov2withNorm',
            encoder_config_path='facebook/dinov2-base',
            encoder_input_size=224,
            encoder_params={'dinov2_path': 'facebook/dinov2-base', 'normalize': True},
            decoder_config_path='facebook/vit-mae-base',
            decoder_patch_size=16,
            pretrained_decoder_path=None,
            num_embeddings=512,
            commitment_cost=0.25,
            vq_decay=0.99,
            noise_tau=0.0,
            reshape_to_2d=True,
            normalization_stat_path=None,
        )
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Test encode
        print("  Testing encode...")
        with torch.no_grad():
            z_q, indices = model.encode(x, return_indices=True)
        print(f"  ✓ Encoded shape: {z_q.shape}")
        print(f"  ✓ Indices shape: {indices.shape}")
        
        # Test decode
        print("  Testing decode...")
        with torch.no_grad():
            x_rec = model.decode(z_q)
        print(f"  ✓ Decoded shape: {x_rec.shape}")
        assert x.shape == x_rec.shape, "Decoded shape should match input"
        
        # Test encode_to_indices and decode_from_indices
        print("  Testing index-based encode/decode...")
        with torch.no_grad():
            indices = model.encode_to_indices(x)
            x_rec_from_indices = model.decode_from_indices(indices)
        print(f"  ✓ Reconstruction from indices shape: {x_rec_from_indices.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed encode/decode test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_quantizer():
    """Test VectorQuantizer module directly."""
    print("\nTest 4: VectorQuantizer Module")
    try:
        from stage1.vector_quantizer import VectorQuantizer
        
        vq = VectorQuantizer(
            num_embeddings=512,
            embedding_dim=768,
            commitment_cost=0.25,
            decay=0.99,
            epsilon=1e-5,
        )
        
        # Test with 2D input (B, C, H, W)
        print("  Testing 2D input format (B, C, H, W)...")
        z_2d = torch.randn(2, 768, 14, 14)
        z_q_2d, loss_2d, indices_2d = vq(z_2d)
        print(f"  ✓ Input shape: {z_2d.shape}")
        print(f"  ✓ Output shape: {z_q_2d.shape}")
        print(f"  ✓ Indices shape: {indices_2d.shape}")
        print(f"  ✓ Loss: {loss_2d.item():.4f}")
        assert z_2d.shape == z_q_2d.shape, "Output should match input shape"
        
        # Test with 1D input (B, N, C)
        print("  Testing 1D input format (B, N, C)...")
        z_1d = torch.randn(2, 196, 768)
        z_q_1d, loss_1d, indices_1d = vq(z_1d)
        print(f"  ✓ Input shape: {z_1d.shape}")
        print(f"  ✓ Output shape: {z_q_1d.shape}")
        print(f"  ✓ Indices shape: {indices_1d.shape}")
        print(f"  ✓ Loss: {loss_1d.item():.4f}")
        assert z_1d.shape == z_q_1d.shape, "Output should match input shape"
        
        # Test get_codebook_entry
        print("  Testing codebook entry retrieval...")
        entries = vq.get_codebook_entry(indices_2d)
        print(f"  ✓ Retrieved entries shape: {entries.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed VectorQuantizer test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("VQRAE Implementation Tests")
    print("="*60)
    
    tests = [
        test_vector_quantizer,
        test_vqrae_initialization,
        test_vqrae_forward,
        test_vqrae_encode_decode,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
