#!/usr/bin/env python3
"""
Simple test script for FSQ (Finite Scalar Quantization) implementation.

This script validates the basic functionality of the FSQ quantizer:
1. Forward pass with 2D inputs (B, C, H, W)
2. Forward pass with 1D inputs (B, N, C)
3. Index encoding and decoding
4. Codebook lookup
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stage1.fsq import FSQ


def test_fsq_basic():
    """Test basic FSQ functionality."""
    print("=" * 60)
    print("Testing FSQ Basic Functionality")
    print("=" * 60)
    
    # Test with small FSQ configuration
    levels = [8, 6, 5]  # 8*6*5 = 240 codes, 3 dimensions
    fsq = FSQ(levels=levels)
    
    print(f"\nFSQ Configuration:")
    print(f"  Levels: {levels}")
    print(f"  Dimension: {fsq.dim}")
    print(f"  Codebook size: {fsq.codebook_size}")
    
    # Test 1: Forward pass with 2D input (B, C, H, W)
    print("\n" + "-" * 60)
    print("Test 1: 2D Input (B, C, H, W)")
    print("-" * 60)
    
    batch_size = 2
    height, width = 4, 4
    z_2d = torch.randn(batch_size, 3, height, width)  # (B, C, H, W)
    print(f"Input shape: {z_2d.shape}")
    
    z_q_2d, loss_2d, indices_2d = fsq(z_2d)
    
    print(f"Quantized shape: {z_q_2d.shape}")
    print(f"Loss: {loss_2d.item():.6f}")
    print(f"Indices shape: {indices_2d.shape}")
    print(f"Indices range: [{indices_2d.min().item()}, {indices_2d.max().item()}]")
    
    # Verify output shape matches input
    assert z_q_2d.shape == z_2d.shape, "Output shape mismatch!"
    assert indices_2d.shape == (batch_size, height, width), "Indices shape mismatch!"
    assert indices_2d.min() >= 0 and indices_2d.max() < fsq.codebook_size, "Invalid indices!"
    print("✓ 2D input test passed")
    
    # Test 2: Forward pass with 1D input (B, N, C)
    print("\n" + "-" * 60)
    print("Test 2: 1D Input (B, N, C)")
    print("-" * 60)
    
    num_tokens = 16
    z_1d = torch.randn(batch_size, num_tokens, 3)  # (B, N, C)
    print(f"Input shape: {z_1d.shape}")
    
    z_q_1d, loss_1d, indices_1d = fsq(z_1d)
    
    print(f"Quantized shape: {z_q_1d.shape}")
    print(f"Loss: {loss_1d.item():.6f}")
    print(f"Indices shape: {indices_1d.shape}")
    print(f"Indices range: [{indices_1d.min().item()}, {indices_1d.max().item()}]")
    
    # Verify output shape matches input
    assert z_q_1d.shape == z_1d.shape, "Output shape mismatch!"
    assert indices_1d.shape == (batch_size, num_tokens), "Indices shape mismatch!"
    assert indices_1d.min() >= 0 and indices_1d.max() < fsq.codebook_size, "Invalid indices!"
    print("✓ 1D input test passed")
    
    # Test 3: Codebook entry retrieval
    print("\n" + "-" * 60)
    print("Test 3: Codebook Entry Retrieval")
    print("-" * 60)
    
    # Get codebook entries for indices
    entries_2d = fsq.get_codebook_entry(indices_2d)
    print(f"Entries shape (2D): {entries_2d.shape}")
    assert entries_2d.shape == (batch_size, height, width, 3), "Entries shape mismatch!"
    
    entries_1d = fsq.get_codebook_entry(indices_1d)
    print(f"Entries shape (1D): {entries_1d.shape}")
    assert entries_1d.shape == (batch_size, num_tokens, 3), "Entries shape mismatch!"
    print("✓ Codebook entry retrieval test passed")
    
    # Test 4: Round-trip encoding/decoding
    print("\n" + "-" * 60)
    print("Test 4: Round-trip Encoding/Decoding")
    print("-" * 60)
    
    # Encode to indices and decode back
    indices_test = torch.randint(0, fsq.codebook_size, (batch_size, height, width))
    entries_decoded = fsq.get_codebook_entry(indices_test)
    indices_reencoded = fsq.codes_to_indices(entries_decoded)
    
    print(f"Original indices shape: {indices_test.shape}")
    print(f"Decoded entries shape: {entries_decoded.shape}")
    print(f"Re-encoded indices shape: {indices_reencoded.shape}")
    
    # Verify round-trip consistency
    assert torch.all(indices_test == indices_reencoded), "Round-trip encoding/decoding failed!"
    print("✓ Round-trip encoding/decoding test passed")
    
    # Test 5: Verify quantization levels
    print("\n" + "-" * 60)
    print("Test 5: Verify Quantization Levels")
    print("-" * 60)
    
    # Create input at exact quantization levels
    z_exact = torch.tensor([
        [[[-3.5, -2.5, -1.5, -0.5],
          [0.5, 1.5, 2.5, 3.5],
          [0.0, 1.0, 2.0, 3.0],
          [-3.0, -2.0, -1.0, 0.0]]],
    ] * 3).permute(0, 1, 3, 2).float()  # (1, 3, 4, 4)
    
    z_q_exact, _, _ = fsq(z_exact)
    
    # Verify that exact levels are preserved
    diff = torch.abs(z_exact - z_q_exact).max().item()
    print(f"Max difference from exact levels: {diff:.6f}")
    assert diff < 0.1, "Exact quantization levels not preserved!"
    print("✓ Quantization levels test passed")
    
    print("\n" + "=" * 60)
    print("All FSQ tests passed successfully! ✓")
    print("=" * 60)


def test_fsq_gradients():
    """Test that gradients flow through FSQ (straight-through estimator)."""
    print("\n" + "=" * 60)
    print("Testing FSQ Gradient Flow")
    print("=" * 60)
    
    levels = [8, 8, 8]
    fsq = FSQ(levels=levels)
    
    # Create input with gradients
    z = torch.randn(2, 3, 4, 4, requires_grad=True)
    
    # Forward pass
    z_q, loss, indices = fsq(z)
    
    # Compute a simple loss
    target = torch.randn_like(z_q)
    reconstruction_loss = torch.nn.functional.mse_loss(z_q, target)
    total_loss = reconstruction_loss + loss
    
    # Backward pass
    total_loss.backward()
    
    print(f"Input gradient shape: {z.grad.shape if z.grad is not None else 'None'}")
    print(f"Gradient norm: {z.grad.norm().item():.6f}" if z.grad is not None else "No gradient")
    
    assert z.grad is not None, "Gradients not flowing through FSQ!"
    assert not torch.isnan(z.grad).any(), "NaN gradients detected!"
    assert not torch.isinf(z.grad).any(), "Inf gradients detected!"
    
    print("✓ Gradient flow test passed")
    print("=" * 60)


def test_fsq_different_levels():
    """Test FSQ with different level configurations."""
    print("\n" + "=" * 60)
    print("Testing FSQ with Different Level Configurations")
    print("=" * 60)
    
    configs = [
        [8, 8, 8],      # 512 codes
        [8, 6, 5],      # 240 codes
        [7, 5, 5, 5],   # 875 codes
        [8, 5, 5, 5],   # 1000 codes
    ]
    
    for levels in configs:
        print(f"\nTesting levels: {levels}")
        fsq = FSQ(levels=levels)
        
        dim = len(levels)
        z = torch.randn(2, dim, 4, 4)
        z_q, loss, indices = fsq(z)
        
        print(f"  Codebook size: {fsq.codebook_size}")
        print(f"  Output shape: {z_q.shape}")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Unique indices: {len(torch.unique(indices))}")
        
        assert z_q.shape == z.shape, f"Shape mismatch for levels {levels}"
        assert indices.max() < fsq.codebook_size, f"Invalid indices for levels {levels}"
        print(f"  ✓ Test passed for levels {levels}")
    
    print("\n" + "=" * 60)
    print("All level configuration tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    # Run all tests
    test_fsq_basic()
    test_fsq_gradients()
    test_fsq_different_levels()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! ✓✓✓")
    print("=" * 60)
