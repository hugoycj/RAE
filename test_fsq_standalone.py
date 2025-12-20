#!/usr/bin/env python3
"""
Simple standalone test for FSQ (Finite Scalar Quantization) implementation.
This test imports FSQ directly without dependencies on other modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class FSQ(nn.Module):
    """
    Finite Scalar Quantization layer (copied for standalone testing).
    """
    
    def __init__(
        self,
        levels: List[int],
        eps: float = 1e-3,
    ):
        super().__init__()
        
        self.levels = levels
        self.eps = eps
        self.dim = len(levels)
        
        # Calculate total codebook size
        self.codebook_size = 1
        for level in levels:
            self.codebook_size *= level
        
        # Precompute the implicit codebook
        self._build_codebook()
        
        print(f"Initialized FSQ with levels={levels}, "
              f"dim={self.dim}, codebook_size={self.codebook_size}")
    
    def _build_codebook(self):
        """Build the implicit codebook."""
        all_levels = []
        for L in self.levels:
            level_values = torch.arange(L, dtype=torch.float32) - (L - 1) / 2
            all_levels.append(level_values)
        
        grids = torch.meshgrid(*all_levels, indexing='ij')
        codebook = torch.stack([grid.flatten() for grid in grids], dim=-1)
        self.register_buffer('codebook', codebook)
        
        basis = []
        for i in range(len(self.levels)):
            basis_value = 1
            for j in range(i + 1, len(self.levels)):
                basis_value *= self.levels[j]
            basis.append(basis_value)
        self.register_buffer('basis', torch.tensor(basis, dtype=torch.long))
    
    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize input to the nearest levels."""
        z_q = z.clone()
        for i, L in enumerate(self.levels):
            max_val = (L - 1) / 2
            z_q[..., i] = torch.clamp(z_q[..., i], -max_val - self.eps, max_val + self.eps)
            z_q[..., i] = torch.round(z_q[..., i])
            z_q[..., i] = torch.clamp(z_q[..., i], -max_val, max_val)
        return z_q
    
    def _compute_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        """Compute codebook indices from quantized values."""
        indices = torch.zeros(z_q.shape[:-1], dtype=torch.long, device=z_q.device)
        for i, L in enumerate(self.levels):
            dim_indices = (z_q[..., i] + (L - 1) / 2).long()
            indices += dim_indices * self.basis[i]
        return indices
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through FSQ."""
        input_shape = z.shape
        is_2d_input = len(input_shape) == 4  # (B, C, H, W)
        
        if is_2d_input:
            z = z.permute(0, 2, 3, 1).contiguous()
        
        if z.shape[-1] != self.dim:
            raise ValueError(
                f"Input dimension {z.shape[-1]} does not match FSQ dimension {self.dim}. "
                f"Expected input shape: (B, {self.dim}, H, W) or (B, N, {self.dim})"
            )
        
        z_q = self._quantize(z)
        indices = self._compute_indices(z_q)
        loss = F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        
        if is_2d_input:
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q, loss, indices
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Retrieve codebook entries for given indices."""
        flat_indices = indices.view(-1)
        entries = self.codebook[flat_indices]
        
        if len(indices.shape) == 3:
            B, H, W = indices.shape
            entries = entries.view(B, H, W, self.dim)
        else:
            B, N = indices.shape
            entries = entries.view(B, N, self.dim)
        
        return entries
    
    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert quantized codes to indices."""
        return self._compute_indices(codes)


def test_fsq_basic():
    """Test basic FSQ functionality."""
    print("=" * 60)
    print("Testing FSQ Basic Functionality")
    print("=" * 60)
    
    levels = [8, 6, 5]  # 8*6*5 = 240 codes, 3 dimensions
    fsq = FSQ(levels=levels)
    
    print(f"\nFSQ Configuration:")
    print(f"  Levels: {levels}")
    print(f"  Dimension: {fsq.dim}")
    print(f"  Codebook size: {fsq.codebook_size}")
    
    # Test 1: Forward pass with 2D input
    print("\n" + "-" * 60)
    print("Test 1: 2D Input (B, C, H, W)")
    print("-" * 60)
    
    batch_size = 2
    height, width = 4, 4
    z_2d = torch.randn(batch_size, 3, height, width)
    print(f"Input shape: {z_2d.shape}")
    
    z_q_2d, loss_2d, indices_2d = fsq(z_2d)
    
    print(f"Quantized shape: {z_q_2d.shape}")
    print(f"Loss: {loss_2d.item():.6f}")
    print(f"Indices shape: {indices_2d.shape}")
    print(f"Indices range: [{indices_2d.min().item()}, {indices_2d.max().item()}]")
    
    assert z_q_2d.shape == z_2d.shape, "Output shape mismatch!"
    assert indices_2d.shape == (batch_size, height, width), "Indices shape mismatch!"
    assert indices_2d.min() >= 0 and indices_2d.max() < fsq.codebook_size, "Invalid indices!"
    print("✓ 2D input test passed")
    
    # Test 2: Forward pass with 1D input
    print("\n" + "-" * 60)
    print("Test 2: 1D Input (B, N, C)")
    print("-" * 60)
    
    num_tokens = 16
    z_1d = torch.randn(batch_size, num_tokens, 3)
    print(f"Input shape: {z_1d.shape}")
    
    z_q_1d, loss_1d, indices_1d = fsq(z_1d)
    
    print(f"Quantized shape: {z_q_1d.shape}")
    print(f"Loss: {loss_1d.item():.6f}")
    print(f"Indices shape: {indices_1d.shape}")
    
    assert z_q_1d.shape == z_1d.shape, "Output shape mismatch!"
    assert indices_1d.shape == (batch_size, num_tokens), "Indices shape mismatch!"
    print("✓ 1D input test passed")
    
    # Test 3: Codebook entry retrieval
    print("\n" + "-" * 60)
    print("Test 3: Codebook Entry Retrieval")
    print("-" * 60)
    
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
    
    indices_test = torch.randint(0, fsq.codebook_size, (batch_size, height, width))
    entries_decoded = fsq.get_codebook_entry(indices_test)
    indices_reencoded = fsq.codes_to_indices(entries_decoded)
    
    assert torch.all(indices_test == indices_reencoded), "Round-trip encoding/decoding failed!"
    print("✓ Round-trip encoding/decoding test passed")
    
    print("\n" + "=" * 60)
    print("All FSQ tests passed successfully! ✓")
    print("=" * 60)


def test_fsq_gradients():
    """Test gradient flow through FSQ."""
    print("\n" + "=" * 60)
    print("Testing FSQ Gradient Flow")
    print("=" * 60)
    
    levels = [8, 8, 8]
    fsq = FSQ(levels=levels)
    
    z = torch.randn(2, 3, 4, 4, requires_grad=True)
    z_q, loss, indices = fsq(z)
    
    target = torch.randn_like(z_q)
    reconstruction_loss = torch.nn.functional.mse_loss(z_q, target)
    total_loss = reconstruction_loss + loss
    total_loss.backward()
    
    print(f"Input gradient shape: {z.grad.shape if z.grad is not None else 'None'}")
    print(f"Gradient norm: {z.grad.norm().item():.6f}" if z.grad is not None else "No gradient")
    
    assert z.grad is not None, "Gradients not flowing through FSQ!"
    assert not torch.isnan(z.grad).any(), "NaN gradients detected!"
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
        print(f"  ✓ Test passed for levels {levels}")
    
    print("\n" + "=" * 60)
    print("All level configuration tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_fsq_basic()
    test_fsq_gradients()
    test_fsq_different_levels()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! ✓✓✓")
    print("=" * 60)
