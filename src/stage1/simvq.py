# coding=utf-8
# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
SimVQ: Similarity-based Vector Quantization

This module implements SimVQ (Similarity Vector Quantization) as described in the
VQRAE paper. SimVQ uses cosine similarity with learnable projection matrices to
avoid codebook collapse and improve quantization quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimVQ(nn.Module):
    """
    Similarity-based Vector Quantization (SimVQ).
    
    Unlike standard VQ which uses L2 distance, SimVQ uses cosine similarity
    with learnable projection matrices. This improves stability and avoids
    codebook collapse.
    
    Args:
        num_embeddings (int): Size of the codebook (number of discrete codes).
        embedding_dim (int): Dimensionality of each embedding vector.
        commitment_cost (float): Weight for the commitment loss term.
        use_projection (bool): Whether to use learnable projection matrices.
        projection_dim (int): Dimension of projection space (if None, uses embedding_dim).
        epsilon (float): Small constant for numerical stability.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        use_projection: bool = True,
        projection_dim: int = None,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_projection = use_projection
        self.projection_dim = projection_dim or embedding_dim
        self.epsilon = epsilon
        
        # Initialize codebook with uniform distribution
        self.register_buffer('embedding', torch.empty(num_embeddings, embedding_dim))
        self.embedding.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # Learnable projection matrices (key component of SimVQ)
        if use_projection:
            self.input_proj = nn.Linear(embedding_dim, self.projection_dim, bias=False)
            self.codebook_proj = nn.Linear(embedding_dim, self.projection_dim, bias=False)
            # Initialize projections
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.xavier_uniform_(self.codebook_proj.weight)
        else:
            self.input_proj = None
            self.codebook_proj = None
        
        # EMA for codebook updates (optional, can be disabled for fully learnable codebook)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.clone())
        
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
            # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
            z_flattened = z.view(input_shape[0], input_shape[1], -1).permute(0, 2, 1).contiguous()
        else:
            # Already in (B, N, C) format
            z_flattened = z.contiguous()
        
        # Flatten to (B*N, C) for similarity computation
        flat_input = z_flattened.view(-1, self.embedding_dim)
        
        # Apply projections if enabled (SimVQ key feature)
        if self.use_projection:
            flat_input_proj = self.input_proj(flat_input)
            codebook_proj = self.codebook_proj(self.embedding)
            
            # Normalize for cosine similarity
            flat_input_norm = F.normalize(flat_input_proj, p=2, dim=1)
            codebook_norm = F.normalize(codebook_proj, p=2, dim=1)
        else:
            # Direct cosine similarity without projection
            flat_input_norm = F.normalize(flat_input, p=2, dim=1)
            codebook_norm = F.normalize(self.embedding, p=2, dim=1)
        
        # Compute cosine similarity (higher is better)
        similarity = torch.matmul(flat_input_norm, codebook_norm.t())
        
        # Get nearest codebook entries (highest similarity)
        encoding_indices = torch.argmax(similarity, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize using the codebook
        quantized = torch.matmul(encodings, self.embedding).view(z_flattened.shape)
        
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
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve codebook entries for given indices.
        
        Args:
            indices: Tensor of codebook indices
        
        Returns:
            Codebook entries corresponding to the indices
        """
        return F.embedding(indices, self.embedding)
    
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
