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


class FSQAdapter(nn.Module):
    def __init__(self, in_dim=768, out_dim=5, hidden_dim=512):
        super().__init__()
        # 冻结 Encoder 后，Input Norm 至关重要，因为我们无法调整上游分布
        self.norm = nn.LayerNorm(in_dim)
        
        # 3层 MLP：足够深，能完成“语义”到“坐标”的复杂非线性映射
        # 结构：降维 -> 激活 -> 混合 -> 激活 -> 输出
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # FSQ 灵魂组件
        # self.post_gain = nn.Parameter(torch.ones(out_dim))
        self.post_gain = nn.Parameter(torch.ones(out_dim) * 3.0)

    def forward(self, x):
        # x: (B, 14, 14, 768)
        
        # 1. 归一化 (让冻结特征分布稳定)
        x = self.norm(x)
        
        # 2. MLP 映射 (逐像素处理，利用 ViT 已有的空间信息)
        x = self.net(x)
        
        # 3. 约束到 FSQ 空间
        x = torch.tanh(x) * self.post_gain
        
        return x

class AttnFSQProjector(nn.Module):
    def __init__(self, in_dim=768, out_dim=5, hidden_dim=512, num_layers=4, nhead=8):
        super().__init__()
        
        # 1. 维度适配 (Bottleneck): 768 -> 512
        # 先降维可以显著减少 Transformer 的计算量，同时起到特征压缩的作用
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # 2. 纯粹的 Transformer Encoder
        # 利用 PyTorch 官方高度优化的模块
        # norm_first=True (Pre-Norm) 对深层网络收敛更稳定
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True, 
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 输出映射
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        
        # 4. FSQ 核心组件 (Tanh + Gain)
        self.post_gain = nn.Parameter(torch.ones(out_dim))

    def forward(self, x):
        # x: (B, 14, 14, 768) 来自冻结的 ViT
        B, H, W, C = x.shape
        
        # Flatten: (B, 14, 14, 768) -> (B, 196, 768)
        x = x.view(B, H * W, C)
        
        # Linear Projection
        x = self.input_proj(x)
        
        # Transformer Context Mixing
        # 不需要加 Pos Emb，因为输入特征本身包含了位置语义
        x = self.transformer(x)
        
        # Project to FSQ Latent
        x = self.output_proj(x)
        
        # FSQ Tanh & Gain
        x = torch.tanh(x) * self.post_gain
        
        # Restore Spatial: (B, 196, 5) -> (B, 14, 14, 5)
        return x.view(B, H, W, -1)

class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size=64, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        # 随机初始化频率矩阵 B (Gaussian mapping)
        # 保持 B 固定不更新，或者作为 Parameter 更新均可，通常固定效果就很好
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
        
    def forward(self, x):
        # x: (..., input_dim) -> (..., mapping_size)
        # 投影到高维频率空间
        x_proj = (2.0 * torch.pi * x) @ self.B
        # 拼接 sin 和 cos，输出维度变为 2 * mapping_size
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierFSQUnprojector(nn.Module):
    def __init__(self, fsq_dim, latent_dim, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = latent_dim
            
        # 1. Fourier Mapping: 把低维坐标 (e.g., 6) 炸开到高维 (e.g., 256)
        fourier_dim = 128
        self.fourier = FourierFeatureMapping(fsq_dim, mapping_size=fourier_dim, scale=10.0)
        input_dim = fourier_dim * 2  # sin + cos
        
        # 2. ResMLP Block: 增加深度和非线性
        # 相比简单的 Linear-GELU-Linear，残差连接允许更深的梯度传播
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            # 这里的残差结构通常在 block 内部做，或者我们可以简单堆叠几个 Dense 层
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_q):
        # z_q: FSQ 输出的离散坐标 (B, ..., fsq_dim)
        
        # 1. Coordinate Encoding
        z_features = self.fourier(z_q)
        
        # 2. Deep Projection
        out = self.mlp(z_features)
        
        return out

class FSQ(nn.Module):
    """
    Fixed Finite Scalar Quantization layer.
    Correctly handles:
    1. Even levels (half-integer codebook values).
    2. Scaling mismatch in Loss computation.
    3. Gradient scaling in STE.
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
        
        # Build implicit codebook
        self._build_codebook()

        # Precompute scales for Tanh input [-1, 1] -> [-(L-1)/2, (L-1)/2]
        scales = []
        for L in levels:
            scales.append((L - 1) / 2)
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))
    
    def _build_codebook(self):
        all_levels = []
        for L in self.levels:
            # Codebook values can be half-integers if L is even (e.g., -0.5, 0.5)
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
        """
        Quantize input to the nearest levels.
        Correctly handles both integer and half-integer levels.
        """
        z_q = z.clone()
        for i, L in enumerate(self.levels):
            # 1. Define valid range for this dimension
            half_width = (L - 1) / 2
            
            # 2. Shift to [0, L-1] domain to simplify quantization
            # This turns half-integers (if any) into integers for rounding
            val = z_q[..., i] + half_width
            
            # 3. Clamp to valid range (with epsilon for safety)
            val = torch.clamp(val, 0 - self.eps, (L - 1) + self.eps)
            
            # 4. Round to nearest integer index
            val = torch.round(val)
            
            # 5. Hard clamp to ensure indices are valid
            val = torch.clamp(val, 0, L - 1)
            
            # 6. Shift back to centered domain
            z_q[..., i] = val - half_width
        
        return z_q
    
    def _compute_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q is already quantized to valid levels, so we just shift and sum
        indices = torch.zeros(z_q.shape[:-1], dtype=torch.long, device=z_q.device)
        for i, L in enumerate(self.levels):
            # Shift from centered value to index [0, L-1]
            # using round() to be safe against float precision errors
            dim_indices = torch.round(z_q[..., i] + (L - 1) / 2).long()
            indices += dim_indices * self.basis[i]
        return indices
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = z.shape
        is_2d_input = len(input_shape) == 4
        
        if is_2d_input:
            z = z.permute(0, 2, 3, 1).contiguous() # (B, H, W, D)
        
        # 1. Scale inputs: Tanh [-1, 1] -> FSQ Range
        z_scaled = z * self.scales

        # 2. Quantize in the scaled domain
        z_q = self._quantize(z_scaled)
        
        # 3. Compute indices
        indices = self._compute_indices(z_q)
        
        # 4. Compute Loss
        # FIX: Compute loss between SCALED input and QUANTIZED output
        loss = F.mse_loss(z_q.detach(), z_scaled)
        
        # 5. Straight-through estimator (STE)
        # FIX: Apply STE in the scaled domain so gradients flow through the scaler correctly
        z_q_out = z_scaled + (z_q - z_scaled).detach()
        
        # Reshape back
        if is_2d_input:
            z_q_out = z_q_out.permute(0, 3, 1, 2).contiguous()
        
        return z_q_out, loss, indices
    
    # helper methods remain the same...
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
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
        return self._compute_indices(codes)
    
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return self.get_codebook_entry(indices)
