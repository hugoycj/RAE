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
        use_l2_norm: bool = True,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.legacy = legacy
        self.epsilon = epsilon
        self.use_l2_norm = use_l2_norm

        # Frozen codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim**-0.5)
        # Freeze the codebook
        for p in self.embedding.parameters():
            p.requires_grad = False
        
        # Learnable projection layer (the key component of SimVQ)
        self.embedding_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)

        # <<< NEW: learnable temperature for cosine distance
        self.l2_norm_scale = nn.Parameter(torch.tensor(10.0))
        
        # Tracking
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        
        # For tracking losses (used by training script - must keep gradients)
        self.last_commit_loss = None
        self.last_codebook_loss = None
        # For monitoring only (detached)
        self.last_commit_loss_detached = None
        self.last_codebook_loss_detached = None
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        # Determine input format and flatten if needed

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
        
        if is_2d_input:
            z = z.permute(0, 2, 3, 1).contiguous()
        
        z_flattened = z.view(-1, self.embedding_dim)
        quant_codebook = self.embedding_proj(self.embedding.weight)

        # ----------------------------------------------------
        # MINIMAL MODIFICATION: insert L2-normalized branch
        # ----------------------------------------------------
        if self.use_l2_norm:
            # L2 normalize both z and codebook to unit sphere
            z_norm = F.normalize(z_flattened, p=2, dim=-1)
            codebook_norm = F.normalize(quant_codebook, p=2, dim=-1)

            # Negative cosine similarity distance
            distances = -torch.einsum('bd,nd->bn', z_norm, codebook_norm) * self.l2_norm_scale

        else:
            # Original Euclidean distance
            distances = (
                torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
                torch.sum(quant_codebook ** 2, dim=1) -
                2 * torch.matmul(z_flattened, quant_codebook.t())
            )
        # ----------------------------------------------------
        
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, quant_codebook).view(z.shape)
        
        commitment_loss = torch.mean((quantized.detach() - z) ** 2)
        codebook_loss = torch.mean((quantized - z.detach()) ** 2)
        
        self.last_commit_loss = commitment_loss
        self.last_codebook_loss = codebook_loss
        self.last_commit_loss_detached = commitment_loss.detach()
        self.last_codebook_loss_detached = codebook_loss.detach()
        
        if not self.legacy:
            vq_loss = self.commitment_cost * commitment_loss + codebook_loss
        else:
            vq_loss = commitment_loss + self.commitment_cost * codebook_loss
        
        quantized = z + (quantized - z).detach()
        
        if is_2d_input:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            encoding_indices = encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])
        else:
            encoding_indices = encoding_indices.view(input_shape[0], input_shape[1])
        
        return quantized, vq_loss, encoding_indices
