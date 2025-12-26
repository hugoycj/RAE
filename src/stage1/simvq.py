# coding=utf-8
# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
SimVQ: Similarity-based Vector Quantization

This module implements SimVQ as described in "SimVQ: Addressing Representation 
Collapse in Vector Quantized Models with One Linear Layer" (arXiv:2411.02038).

SimVQ uses a frozen codebook with a learnable linear projection to avoid 
codebook collapse while achieving 100% codebook utilization.

Reference implementation: https://github.com/youngsheen/SimVQ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SimVQ(nn.Module):
    """
    SimVQ: Vector Quantization with frozen codebook and learnable projection.
    
    Key idea: Instead of learning the codebook directly, SimVQ keeps the codebook
    frozen and learns a single linear projection layer. This simple modification
    achieves 100% codebook utilization and prevents representation collapse.
    
    Args:
        num_embeddings (int): Size of the codebook (number of discrete codes).
        embedding_dim (int): Dimensionality of each embedding vector.
        commitment_cost (float): Weight for the commitment loss term (beta).
        legacy (bool): If True, uses legacy loss formulation (for backwards compatibility).
        epsilon (float): Small constant for numerical stability.
        use_l2_norm (bool): If True, use L2-normalized cosine similarity distance instead of Euclidean.
        sane_index_shape (bool): Parameter kept for API compatibility with reference implementation.
            Currently, indices are always reshaped to (batch, height, width) for VQRAE compatibility.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        legacy: bool = True,
        epsilon: float = 1e-5,
        use_l2_norm: bool = True,
        sane_index_shape: bool = False,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.legacy = legacy
        self.epsilon = epsilon
        self.use_l2_norm = use_l2_norm
        self.sane_index_shape = sane_index_shape

        # Frozen codebook - initialized with normal distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim**-0.5)
        # Freeze the codebook
        for p in self.embedding.parameters():
            p.requires_grad = False
        
        # Learnable projection layer (the key component of SimVQ)
        # Using default bias=True to match reference implementation
        self.embedding_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)

        # Learnable temperature for L2-normalized cosine distance (only if using L2 norm)
        if use_l2_norm:
            self.l2_norm_scale = nn.Parameter(torch.tensor(10.0))
        else:
            self.l2_norm_scale = None
        
        # For tracking losses (used by training script - must keep gradients)
        self.last_commit_loss = None
        self.last_codebook_loss = None
        # For monitoring only (detached)
        self.last_commit_loss_detached = None
        self.last_codebook_loss_detached = None
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for SimVQ quantization.

        Args:
            z: Input tensor of shape (batch_size, embedding_dim, height, width)
               or (batch_size, num_tokens, embedding_dim)
        
        Returns:
            quantized: Quantized tensor with same shape as input
            vq_loss: Vector quantization loss (commitment + codebook loss)
            encoding_indices: Indices of selected codebook entries
        """
        # Determine input format and flatten if needed
        input_shape = z.shape
        is_2d_input = len(input_shape) == 4
        
        # Convert to (batch, height, width, channel) format for consistency with reference
        if is_2d_input:
            z = z.permute(0, 2, 3, 1).contiguous()
        
        # Flatten to (N, embedding_dim) for distance computation
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Apply learned projection to frozen codebook
        quant_codebook = self.embedding_proj(self.embedding.weight)

        # Compute distances based on distance metric
        if self.use_l2_norm:
            # L2-normalized cosine similarity distance
            # Normalize both z and codebook to unit sphere
            z_norm = F.normalize(z_flattened, p=2, dim=-1)
            codebook_norm = F.normalize(quant_codebook, p=2, dim=-1)
            # Negative cosine similarity (with learnable temperature)
            # Using @ operator for better performance
            distances = -(z_norm @ codebook_norm.t()) * self.l2_norm_scale
        else:
            # Euclidean distance: (z - e)^2 = z^2 + e^2 - 2*z*e
            # z_flattened: (batch*spatial, embedding_dim)
            # quant_codebook: (num_embeddings, embedding_dim)
            distances = (
                torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
                torch.sum(quant_codebook ** 2, dim=1) -
                2 * torch.matmul(z_flattened, quant_codebook.t())
            )
        
        # Find nearest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Get quantized values
        quantized = F.embedding(encoding_indices, quant_codebook).view(z.shape)
        
        # Compute commitment and codebook losses
        commitment_loss = torch.mean((quantized.detach() - z) ** 2)
        codebook_loss = torch.mean((quantized - z.detach()) ** 2)
        
        # Store losses for tracking (with and without gradients)
        self.last_commit_loss = commitment_loss
        self.last_codebook_loss = codebook_loss
        self.last_commit_loss_detached = commitment_loss.detach()
        self.last_codebook_loss_detached = codebook_loss.detach()
        
        # Combine losses based on legacy flag
        if not self.legacy:
            vq_loss = self.commitment_cost * commitment_loss + codebook_loss
        else:
            vq_loss = commitment_loss + self.commitment_cost * codebook_loss
        
        # Straight-through estimator: preserve gradients
        quantized = z + (quantized - z).detach()
        
        # Reshape back to original format
        if is_2d_input:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            # Always reshape indices for 2D inputs for VQRAE compatibility
            # The sane_index_shape parameter is kept for reference implementation compatibility
            encoding_indices = encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])
        else:
            encoding_indices = encoding_indices.view(input_shape[0], input_shape[1])
        
        return quantized, vq_loss, encoding_indices
    
    def get_codebook_entry(self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        Get quantized latent vectors from codebook indices.
        
        Args:
            indices: Codebook indices of shape (batch, height, width) or (batch, num_tokens)
            shape: Optional shape for compatibility with reference implementation.
                   Expected format: (batch, height, width, channel) where channel is last dimension.
                   If provided, output will be reshaped to (batch, channel, height, width).
                   If not provided, output shape is inferred from indices shape.
        
        Returns:
            z_q: Quantized latent vectors. 
                 - If shape is provided: returns (batch, channel, height, width)
                 - If shape is None and indices are 3D: returns (batch, height, width, channel)
                 - If shape is None and indices are 2D: returns (batch, num_tokens, channel)
        """
        # Apply projection to frozen codebook
        quant_codebook = self.embedding_proj(self.embedding.weight)
        
        # Get quantized vectors from projected codebook
        z_q = F.embedding(indices, quant_codebook)
        
        # If shape is provided (reference implementation compatibility)
        if shape is not None:
            # Validate shape has 4 dimensions as expected
            if len(shape) != 4:
                raise ValueError(f"Shape must have 4 dimensions (batch, height, width, channel), got {len(shape)}")
            # Reshape to (batch, height, width, channel)
            z_q = z_q.view(shape)
            # Convert to (batch, channel, height, width) - channel moves from last to second position
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q
