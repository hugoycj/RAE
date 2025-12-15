# coding=utf-8
# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
Vector Quantization module for VQRAE.

This module implements vector quantization as used in VQ-VAE and related models.
The quantizer learns a codebook of embeddings and maps continuous latent vectors
to discrete codes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer that discretizes continuous latent representations.
    
    This implements the core VQ mechanism from "Neural Discrete Representation Learning"
    (van den Oord et al., 2017) with exponential moving average updates.
    
    Args:
        num_embeddings (int): Size of the codebook (number of discrete codes).
        embedding_dim (int): Dimensionality of each embedding vector.
        commitment_cost (float): Weight for the commitment loss term.
        decay (float): Decay rate for exponential moving average updates.
        epsilon (float): Small constant for numerical stability.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize codebook with uniform distribution
        self.register_buffer('embedding', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.clone())
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the vector quantizer.
        
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
            # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
            z_flattened = z.view(input_shape[0], input_shape[1], -1).permute(0, 2, 1).contiguous()
        else:
            # Already in (B, N, C) format
            z_flattened = z.contiguous()
        
        # Flatten to (B*N, C) for distance computation
        flat_input = z_flattened.view(-1, self.embedding_dim)
        
        # Calculate distances to codebook entries
        # distances[i, j] = ||z_i - e_j||^2
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.t())
        )
        
        # Get nearest codebook entries
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize using the codebook
        quantized = torch.matmul(encodings, self.embedding).view(z_flattened.shape)
        
        # Update codebook using exponential moving average (only during training)
        if self.training:
            self._ema_update(flat_input, encodings)
        
        # Calculate losses
        # Commitment loss: encourages encoder output to stay close to chosen codebook entry
        e_latent_loss = F.mse_loss(quantized.detach(), z_flattened)
        # Codebook loss: encourages codebook entries to move closer to encoder outputs
        q_latent_loss = F.mse_loss(quantized, z_flattened.detach())
        
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator: copy gradients from decoder to encoder
        quantized = z_flattened + (quantized - z_flattened).detach()
        
        # Reshape back to input format
        if is_2d_input:
            # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
            quantized = quantized.permute(0, 2, 1).contiguous()
            quantized = quantized.view(input_shape)
            encoding_indices = encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])
        else:
            # Already in correct (B, N, C) format
            encoding_indices = encoding_indices.view(input_shape[0], input_shape[1])
        
        return quantized, vq_loss, encoding_indices
    
    def _ema_update(self, flat_input: torch.Tensor, encodings: torch.Tensor):
        """
        Update codebook using exponential moving average.
        
        Args:
            flat_input: Flattened input tensor (B*N, C)
            encodings: One-hot encoded assignments (B*N, num_embeddings)
        """
        # Update cluster sizes
        self.ema_cluster_size.mul_(self.decay).add_(
            encodings.sum(0), alpha=1 - self.decay
        )
        
        # Laplace smoothing of cluster sizes
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon)
            * n
        )
        
        # Update embeddings
        dw = torch.matmul(encodings.t(), flat_input)
        self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        
        self.embedding.copy_(self.ema_w / cluster_size.unsqueeze(1))
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve codebook entries for given indices.
        
        Args:
            indices: Tensor of codebook indices
        
        Returns:
            Codebook entries corresponding to the indices
        """
        # indices shape: (batch, spatial_dim1, spatial_dim2, ...)
        # output shape: (batch, spatial_dim1, spatial_dim2, ..., embedding_dim)
        return F.embedding(indices, self.embedding)
