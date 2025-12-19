# coding=utf-8
# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
VQRAE: Vector Quantized Representation Autoencoder

This module extends RAE with vector quantization to enable discrete latent representations.
The quantized latents can be used for improved controllability and interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rae import RAE
from .vector_quantizer import VectorQuantizer
from typing import Optional, Tuple, Dict, Union


class VQRAE(RAE):
    """
    Vector Quantized Representation Autoencoder.
    
    Extends the standard RAE with vector quantization in the latent space.
    This enables discrete representations while maintaining the pretrained encoder
    and learned decoder from RAE.
    
    Args:
        num_embeddings (int): Size of the codebook for vector quantization.
        commitment_cost (float): Weight for the commitment loss in VQ.
        vq_decay (float): Decay rate for EMA updates in VQ.
        vq_epsilon (float): Small constant for numerical stability in VQ.
        quantize_before_reshape (bool): If True, quantize in (B, N, C) format before
            reshaping to 2D. If False, quantize after reshaping to (B, C, H, W).
        **rae_kwargs: Additional arguments passed to RAE parent class.
    """
    
    def __init__(
        self,
        # Vector quantization specific parameters
        num_embeddings: int = 16384,  # Paper uses 16k for 100% utilization
        commitment_cost: float = 0.25,
        vq_decay: float = 0.99,
        vq_epsilon: float = 1e-5,
        quantize_before_reshape: bool = False,
        use_simvq: bool = False,  # Use SimVQ (set True for paper-compliant implementation)
        # RAE parameters (passed to parent)
        encoder_cls: str = 'SigLIP2wNorm',  # Paper uses SigLIP-L
        encoder_config_path: str = 'google/siglip-large-patch16-384',
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        noise_tau: float = 0.0,  # Paper doesn't use noise
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        freeze_encoder: bool = True,  # NEW: Support for Stage-2 unfreezing
    ):
        # Initialize parent RAE
        super().__init__(
            encoder_cls=encoder_cls,
            encoder_config_path=encoder_config_path,
            encoder_input_size=encoder_input_size,
            encoder_params=encoder_params,
            decoder_config_path=decoder_config_path,
            decoder_patch_size=decoder_patch_size,
            pretrained_decoder_path=pretrained_decoder_path,
            noise_tau=noise_tau,
            reshape_to_2d=reshape_to_2d,
            normalization_stat_path=normalization_stat_path,
            eps=eps,
        )
        
        # Initialize vector quantizer (SimVQ or standard VQ)
        self.num_embeddings = num_embeddings
        self.quantize_before_reshape = quantize_before_reshape
        self.use_simvq = use_simvq
        self.freeze_encoder = freeze_encoder
        
        if use_simvq:
            from .simvq import SimVQ
            self.quantizer = SimVQ(
                num_embeddings=num_embeddings,
                embedding_dim=self.latent_dim,
                commitment_cost=commitment_cost,
                epsilon=vq_epsilon,
            )
            print(f"Initialized VQRAE with SimVQ (codebook size: {num_embeddings}, dim: {self.latent_dim})")
        else:
            from .vector_quantizer import VectorQuantizer
            self.quantizer = VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=self.latent_dim,
                commitment_cost=commitment_cost,
                decay=vq_decay,
                epsilon=vq_epsilon,
            )
            print(f"Initialized VQRAE with standard VQ (codebook size: {num_embeddings})")
        
        # Freeze/unfreeze encoder based on training stage
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen (Stage-1 mode)")
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True
            print("Encoder unfrozen (Stage-2 mode)")
        
        # Store last VQ loss for monitoring
        self.last_vq_loss = None
        self.last_continuous_features = None  # For Stage-2 distillation
        
        print(f"Initialized VQRAE with codebook size {num_embeddings}")
    
    def encode(self, x: torch.Tensor, return_indices: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input to quantized latent representation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_indices: If True, also return the codebook indices
        
        Returns:
            z_q: Quantized latent representation
            indices: (optional) Codebook indices if return_indices=True
        """
        # Get continuous latent from parent encoder
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(
                x, size=(self.encoder_input_size, self.encoder_input_size), 
                mode='bicubic', align_corners=False
            )
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        z = self.encoder(x)
        
        # Apply noise during training if enabled
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        
        # Quantize before or after reshaping based on configuration
        if self.quantize_before_reshape:
            # Quantize in (B, N, C) format
            z_q, vq_loss, indices = self.quantizer(z)
            
            # Then reshape to 2D if needed
            if self.reshape_to_2d:
                b, n, c = z_q.shape
                h = w = int(n ** 0.5)
                z_q = z_q.transpose(1, 2).view(b, c, h, w)
        else:
            # Reshape first if needed
            if self.reshape_to_2d:
                b, n, c = z.shape
                h = w = int(n ** 0.5)
                z = z.transpose(1, 2).view(b, c, h, w)
            
            # Then quantize in (B, C, H, W) format
            z_q, vq_loss, indices = self.quantizer(z)
        
        # Store VQ loss for monitoring
        self.last_vq_loss = vq_loss.detach() if vq_loss is not None else None
        
        # Apply normalization if enabled
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z_q.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z_q.device) if self.latent_var is not None else 1
            z_q = (z_q - latent_mean) / torch.sqrt(latent_var + self.eps)
        
        if return_indices:
            return z_q, indices
        return z_q
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latent to image.
        
        Args:
            z_q: Quantized latent representation
        
        Returns:
            x_rec: Reconstructed image
        """
        # Use parent's decode method
        return super().decode(z_q)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_loss: bool = False,
        return_indices: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through VQRAE.
        
        Args:
            x: Input image tensor
            return_loss: If True, return reconstruction and VQ losses
            return_indices: If True, return codebook indices
        
        Returns:
            If return_loss=False and return_indices=False:
                x_rec: Reconstructed image
            If return_loss=True:
                (x_rec, losses): Reconstructed image and dictionary of losses
            If return_indices=True:
                (x_rec, indices): Reconstructed image and codebook indices
        
        Note:
            If both return_loss and return_indices are True, return_loss takes precedence.
        """
        # Get continuous latent from encoder (before quantization)
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x_normalized = nn.functional.interpolate(
                x, size=(self.encoder_input_size, self.encoder_input_size),
                mode='bicubic', align_corners=False
            )
        else:
            x_normalized = x
        
        x_normalized = (x_normalized - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        z = self.encoder(x_normalized)
        
        # Apply noise during training if enabled
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        
        # Quantize
        if self.quantize_before_reshape:
            z_q, vq_loss, indices = self.quantizer(z)
            if self.reshape_to_2d:
                b, n, c = z_q.shape
                h_lat = w_lat = int(n ** 0.5)
                assert h_lat * w_lat == n, f"Number of patches {n} must be a perfect square for 2D reshaping"
                z_q = z_q.transpose(1, 2).view(b, c, h_lat, w_lat)
        else:
            if self.reshape_to_2d:
                b, n, c = z.shape
                h_lat = w_lat = int(n ** 0.5)
                assert h_lat * w_lat == n, f"Number of patches {n} must be a perfect square for 2D reshaping"
                z = z.transpose(1, 2).view(b, c, h_lat, w_lat)
            z_q, vq_loss, indices = self.quantizer(z)
        
        # Apply normalization
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z_q.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z_q.device) if self.latent_var is not None else 1
            z_q = (z_q - latent_mean) / torch.sqrt(latent_var + self.eps)
        
        # Decode
        x_rec = self.decode(z_q)
        
        if return_loss:
            # Calculate reconstruction loss
            recon_loss = nn.functional.mse_loss(x_rec, x)
            losses = {
                'recon_loss': recon_loss,
                'vq_loss': vq_loss,
                'total_loss': recon_loss + vq_loss
            }
            return x_rec, losses
        
        if return_indices:
            return x_rec, indices
        
        return x_rec
    
    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to codebook indices.
        
        Args:
            x: Input image tensor
        
        Returns:
            indices: Codebook indices
        """
        _, indices = self.encode(x, return_indices=True)
        return indices
    
    def decode_from_indices(self, indices: torch.Tensor, spatial_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Decode from codebook indices to image.
        
        Args:
            indices: Codebook indices
            spatial_shape: Optional (height, width) of latent spatial dimensions.
                          If None, inferred from indices shape.
        
        Returns:
            x_rec: Reconstructed image
        """
        # Get embeddings from indices
        z_q = self.quantizer.get_codebook_entry(indices)
        
        # Reshape to expected format
        if self.quantize_before_reshape:
            # z_q is in (B, H, W, C) format from get_codebook_entry
            # Need to convert to (B, N, C) then potentially to (B, C, H, W)
            b, h, w, c = z_q.shape
            z_q = z_q.view(b, h * w, c)
            
            if self.reshape_to_2d:
                z_q = z_q.transpose(1, 2).view(b, c, h, w)
        else:
            # z_q is in (B, H, W, C) format, need to convert to (B, C, H, W)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # Apply normalization if enabled
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z_q.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z_q.device) if self.latent_var is not None else 1
            z_q = (z_q - latent_mean) / torch.sqrt(latent_var + self.eps)
        
        # Decode
        return self.decode(z_q)
    
    def compute_distillation_loss(self, continuous_features: torch.Tensor, quantized_features: torch.Tensor) -> torch.Tensor:
        """
        Compute self-distillation loss for Stage-2 training.
        
        This loss encourages the quantized features to match the continuous features,
        preserving the semantic understanding from the frozen encoder (Stage-1).
        
        Args:
            continuous_features: Continuous latent features from encoder (before quantization)
            quantized_features: Quantized latent features (after VQ)
        
        Returns:
            distillation_loss: L2 loss between continuous and quantized features
        """
        return F.mse_loss(quantized_features, continuous_features.detach())
    
    def set_stage(self, stage: int):
        """
        Set training stage (1 or 2) and adjust encoder freezing accordingly.
        
        Args:
            stage: Training stage (1 = frozen encoder, 2 = unfrozen encoder)
        """
        if stage == 1:
            # Stage-1: Freeze encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.freeze_encoder = True
            print("Switched to Stage-1 mode (encoder frozen)")
        elif stage == 2:
            # Stage-2: Unfreeze encoder
            for param in self.encoder.parameters():
                param.requires_grad = True
            self.freeze_encoder = False
            print("Switched to Stage-2 mode (encoder unfrozen)")
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")
    
    def get_codebook_usage(self) -> float:
        """
        Get codebook utilization rate.
        
        Returns:
            usage: Percentage of codebook entries being used (0.0 to 1.0)
        """
        if hasattr(self.quantizer, 'get_codebook_usage'):
            return self.quantizer.get_codebook_usage()
        return 0.0
