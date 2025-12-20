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
from typing import Tuple


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
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        legacy: bool = True,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost  # beta in the paper
        self.legacy = legacy
        self.epsilon = epsilon
        
        # Frozen codebook - initialized with normal distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim**-0.5)
        # Freeze the codebook
        for p in self.embedding.parameters():
            p.requires_grad = False
        
        # Learnable projection layer (the key component of SimVQ)
        self.embedding_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        
        # For tracking codebook usage
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        
        # For tracking losses (used by training script - must keep gradients)
        self.last_commit_loss = None
        self.last_codebook_loss = None
        # For monitoring only (detached)
        self.last_commit_loss_detached = None
        self.last_codebook_loss_detached = None
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through SimVQ.
        
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
        is_2d_input = len(input_shape) == 4  # (B, C, H, W)
        
        if is_2d_input:
            # Flatten spatial dimensions: (B, C, H, W) -> (B, H, W, C)
            z = z.permute(0, 2, 3, 1).contiguous()
        
        # Flatten to (B*N, C) for distance computation
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Project the codebook using the learnable linear layer
        quant_codebook = self.embedding_proj(self.embedding.weight)
        
        # Compute L2 distances: (z - e)^2 = z^2 + e^2 - 2*z*e
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
            torch.sum(quant_codebook ** 2, dim=1) -
            2 * torch.matmul(z_flattened, quant_codebook.t())
        )
        
        # Get nearest codebook entries (minimum distance)
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantize using the projected codebook
        quantized = F.embedding(encoding_indices, quant_codebook).view(z.shape)
        
        # Compute loss
        # For SimVQ, codebook is frozen, so only commitment loss is meaningful
        # Commitment loss: encourages encoder output to stay close to chosen codebook entry
        commitment_loss = torch.mean((quantized.detach() - z) ** 2)
        
        # Codebook loss would be: torch.mean((quantized - z.detach()) ** 2)
        # But since codebook is frozen (requires_grad=False), this is always 0 in practice
        # We still compute it for compatibility but it won't contribute to gradients
        codebook_loss = torch.mean((quantized - z.detach()) ** 2)
        
        # Store for training (keep gradients for backprop)
        self.last_commit_loss = commitment_loss
        self.last_codebook_loss = codebook_loss
        # Store detached versions for monitoring only
        self.last_commit_loss_detached = commitment_loss.detach()
        self.last_codebook_loss_detached = codebook_loss.detach()
        
        if not self.legacy:
            # Standard formulation: beta * commitment + codebook
            vq_loss = self.commitment_cost * commitment_loss + codebook_loss
        else:
            # Legacy formulation (used by reference SimVQ for backwards compatibility)
            vq_loss = commitment_loss + self.commitment_cost * codebook_loss
        
        # Straight-through estimator: copy gradients from decoder to encoder
        quantized = z + (quantized - z).detach()
        
        # Reshape back to input format
        if is_2d_input:
            # (B, H, W, C) -> (B, C, H, W)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            encoding_indices = encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])
        else:
            # (B, N, C) format
            encoding_indices = encoding_indices.view(input_shape[0], input_shape[1])
        
        return quantized, vq_loss, encoding_indices
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve codebook entries for given indices.
        
        Args:
            indices: Tensor of codebook indices
        
        Returns:
            Codebook entries corresponding to the indices (from projected codebook)
        """
        # Use projected codebook for retrieval
        quant_codebook = self.embedding_proj(self.embedding.weight)
        return F.embedding(indices, quant_codebook)
    
    def get_codebook_usage(self) -> float:
        """
        Calculate the percentage of codebook entries being used.
        
        Returns:
            Usage percentage (0.0 to 1.0)
        """
        if self.ema_cluster_size.sum() == 0:
            return 0.0
        used_codes = (self.ema_cluster_size > 0).sum().item()
        return used_codes / self.num_embeddings
