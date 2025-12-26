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
        sane_index_shape (bool): If True, reshape indices to (batch, height, width) format.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        legacy: bool = True,
        epsilon: float = 1e-5,
        sane_index_shape: bool = False,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.legacy = legacy
        self.epsilon = epsilon
        self.sane_index_shape = sane_index_shape

        # Frozen codebook - initialized with normal distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim**-0.5)
        # Freeze the codebook
        for p in self.embedding.parameters():
            p.requires_grad = False
        
        # Learnable projection layer (the key component of SimVQ)
        self.embedding_proj = nn.Linear(embedding_dim, embedding_dim)
        
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

        # Compute Euclidean distances: (z - e)^2 = z^2 + e^2 - 2*z*e
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
            torch.sum(quant_codebook ** 2, dim=1) -
            2 * torch.einsum('bd,dn->bn', z_flattened, quant_codebook.t())
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
            if self.sane_index_shape:
                encoding_indices = encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])
            else:
                encoding_indices = encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])
        else:
            encoding_indices = encoding_indices.view(input_shape[0], input_shape[1])
        
        return quantized, vq_loss, encoding_indices
    
    def get_codebook_entry(self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        Get quantized latent vectors from codebook indices.
        
        Args:
            indices: Codebook indices, can be flattened or shaped
            shape: Optional shape specifying (batch, height, width, channel).
                   If provided, output will be reshaped and permuted to (batch, channel, height, width).
        
        Returns:
            z_q: Quantized latent vectors from the codebook
        """
        # Apply projection to frozen codebook
        quant_codebook = self.embedding_proj(self.embedding.weight)
        
        # Get quantized vectors from projected codebook
        z_q = F.embedding(indices.view(-1), quant_codebook)
        
        if shape is not None:
            # Reshape to (batch, height, width, channel)
            z_q = z_q.view(shape)
            # Convert to (batch, channel, height, width)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q
