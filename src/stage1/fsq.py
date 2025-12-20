# coding=utf-8
# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
FSQ: Finite Scalar Quantization

This module implements Finite Scalar Quantization as described in 
"Finite Scalar Quantization: VQ-VAE Made Simple" (arXiv:2309.15505).

FSQ replaces the learned codebook in VQ-VAE with a simple quantization scheme
where each dimension is independently quantized to a fixed set of levels.
This eliminates codebook collapse and achieves implicit 100% codebook utilization.

Key advantages:
- No codebook learning required
- No codebook collapse
- Deterministic and simple
- Fast training and inference

Reference: https://arxiv.org/pdf/2309.15505
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class FSQ(nn.Module):
    """
    Finite Scalar Quantization layer.
    
    FSQ quantizes each dimension of the input independently to a fixed set of levels.
    The codebook is implicitly defined by the Cartesian product of all level sets.
    
    For example, with levels=[8, 8, 8], we get 8^3 = 512 implicit codebook entries.
    Each of the 3 dimensions is quantized to one of 8 levels: {-4, -3, -2, -1, 0, 1, 2, 3}.
    
    Args:
        levels (List[int]): Number of quantization levels per dimension.
            The total codebook size is the product of all levels.
            Common configurations:
            - [8, 8, 8] -> 512 codes
            - [8, 6, 5] -> 240 codes
            - [7, 5, 5, 5] -> 875 codes
            - [8, 5, 5, 5] -> 1000 codes
        eps (float): Small constant for numerical stability.
    
    Examples:
        >>> # 512 codes across 3 dimensions
        >>> fsq = FSQ(levels=[8, 8, 8])
        >>> z = torch.randn(4, 3, 16, 16)  # (B, C, H, W)
        >>> z_q, loss, indices = fsq(z)
        
        >>> # 1000 codes across 4 dimensions
        >>> fsq = FSQ(levels=[8, 5, 5, 5])
        >>> z = torch.randn(4, 4, 16, 16)  # (B, C, H, W)
        >>> z_q, loss, indices = fsq(z)
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
        
        # Precompute the implicit codebook for efficient lookups
        # This is the Cartesian product of all level sets
        self._build_codebook()
        
        print(f"Initialized FSQ with levels={levels}, "
              f"dim={self.dim}, codebook_size={self.codebook_size}")
    
    def _build_codebook(self):
        """
        Build the implicit codebook by computing the Cartesian product of level sets.
        
        For levels=[L1, L2, ..., Ld], this creates all possible combinations of
        quantized values across dimensions.
        """
        # Create quantization levels for each dimension
        # For L levels, use values: -(L-1)/2, ..., -1/2, 1/2, ..., (L-1)/2
        # This centers the levels around 0
        all_levels = []
        for L in self.levels:
            # Generate levels from -(L-1) to (L-1) by steps of 2
            # Then divide by 2 to get -(L-1)/2 to (L-1)/2
            level_values = torch.arange(L, dtype=torch.float32) - (L - 1) / 2
            all_levels.append(level_values)
        
        # Create all combinations using meshgrid
        grids = torch.meshgrid(*all_levels, indexing='ij')
        
        # Stack to create codebook of shape (codebook_size, dim)
        codebook = torch.stack([grid.flatten() for grid in grids], dim=-1)
        
        # Register as buffer so it moves with the model
        self.register_buffer('codebook', codebook)
        
        # Precompute basis for index calculation
        # basis[i] is the product of all levels[j] for j > i
        basis = []
        for i in range(len(self.levels)):
            basis_value = 1
            for j in range(i + 1, len(self.levels)):
                basis_value *= self.levels[j]
            basis.append(basis_value)
        self.register_buffer('basis', torch.tensor(basis, dtype=torch.long))
    
    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantize input to the nearest levels.
        
        Args:
            z: Input tensor of shape (..., dim)
        
        Returns:
            z_q: Quantized tensor of same shape as input
        """
        # For each dimension, round to the nearest quantization level
        z_q = z.clone()
        for i, L in enumerate(self.levels):
            # Clamp to valid range: [-(L-1)/2, (L-1)/2]
            max_val = (L - 1) / 2
            z_q[..., i] = torch.clamp(z_q[..., i], -max_val - self.eps, max_val + self.eps)
            # Round to nearest integer level
            z_q[..., i] = torch.round(z_q[..., i])
            # Clamp again to ensure we're exactly at a level
            z_q[..., i] = torch.clamp(z_q[..., i], -max_val, max_val)
        
        return z_q
    
    def _compute_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Compute codebook indices from quantized values.
        
        Args:
            z_q: Quantized tensor of shape (..., dim)
        
        Returns:
            indices: Codebook indices of shape (...)
        """
        # Convert quantized values to indices
        # For each dimension, shift from [-(L-1)/2, (L-1)/2] to [0, L-1]
        indices = torch.zeros(z_q.shape[:-1], dtype=torch.long, device=z_q.device)
        
        for i, L in enumerate(self.levels):
            # Shift to [0, L-1]
            dim_indices = (z_q[..., i] + (L - 1) / 2).long()
            # Add to total index using basis
            indices += dim_indices * self.basis[i]
        
        return indices
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through FSQ.
        
        Args:
            z: Input tensor of shape (batch_size, dim, height, width)
               or (batch_size, num_tokens, dim)
        
        Returns:
            z_q: Quantized tensor with same shape as input
            loss: Quantization loss (commitment loss only)
            indices: Codebook indices
        """
        # Determine input format and reshape if needed
        input_shape = z.shape
        is_2d_input = len(input_shape) == 4  # (B, C, H, W)
        
        if is_2d_input:
            # Reshape to (B, H, W, C)
            z = z.permute(0, 2, 3, 1).contiguous()
        
        # Verify dimension matches
        if z.shape[-1] != self.dim:
            raise ValueError(
                f"Input dimension {z.shape[-1]} does not match FSQ dimension {self.dim}. "
                f"Expected input shape: (B, {self.dim}, H, W) or (B, N, {self.dim})"
            )
        
        # Quantize
        z_q = self._quantize(z)
        
        # Compute indices
        indices = self._compute_indices(z_q)
        
        # Compute loss (commitment loss only, similar to VQ-VAE)
        # This encourages the encoder to output values close to quantization levels
        loss = F.mse_loss(z_q.detach(), z)
        
        # Straight-through estimator: pass gradients through
        z_q = z + (z_q - z).detach()
        
        # Reshape back to input format
        if is_2d_input:
            # (B, H, W, C) -> (B, C, H, W)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q, loss, indices
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve codebook entries for given indices.
        
        Args:
            indices: Tensor of codebook indices of shape (B, H, W) or (B, N)
        
        Returns:
            Codebook entries of shape (B, H, W, dim) or (B, N, dim)
        """
        # Flatten indices for lookup
        flat_indices = indices.view(-1)
        
        # Look up in codebook
        entries = self.codebook[flat_indices]
        
        # Reshape to match input
        if len(indices.shape) == 3:
            # (B, H, W) case
            B, H, W = indices.shape
            entries = entries.view(B, H, W, self.dim)
        else:
            # (B, N) case
            B, N = indices.shape
            entries = entries.view(B, N, self.dim)
        
        return entries
    
    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert quantized codes to indices.
        
        Args:
            codes: Quantized codes of shape (..., dim)
        
        Returns:
            indices: Codebook indices of shape (...)
        """
        return self._compute_indices(codes)
    
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert indices to quantized codes.
        
        Args:
            indices: Codebook indices
        
        Returns:
            codes: Quantized codes
        """
        return self.get_codebook_entry(indices)
