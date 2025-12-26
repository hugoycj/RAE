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
    
    Supports three quantization methods:
    1. Standard VQ-VAE with EMA codebook updates
    2. SimVQ with frozen codebook and learned projection
    3. FSQ (Finite Scalar Quantization) with implicit codebook
    
    Args:
        num_embeddings (int): Size of the codebook for VQ/SimVQ (ignored for FSQ).
        commitment_cost (float): Weight for the commitment loss in VQ.
        vq_decay (float): Decay rate for EMA updates in standard VQ.
        vq_epsilon (float): Small constant for numerical stability.
        quantize_before_reshape (bool): If True, quantize in (B, N, C) format before
            reshaping to 2D. If False, quantize after reshaping to (B, C, H, W).
        use_simvq (bool): Use SimVQ instead of standard VQ.
        use_fsq (bool): Use FSQ (Finite Scalar Quantization) instead of VQ/SimVQ.
        fsq_levels (List[int]): FSQ quantization levels per dimension (e.g., [8, 8, 8]).
            If use_fsq=True, this defines the quantization levels. The product determines
            the effective codebook size.
        fsq_use_projection (bool): If True (default), uses learned projection layers to
            map between encoder latent dimension and FSQ dimension. This allows using
            low-dimensional FSQ (e.g., 3-6 dims) with high-dimensional encoders (e.g., 768 dims).
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
        use_fsq: bool = False,  # Use FSQ (Finite Scalar Quantization)
        fsq_levels: Optional[list] = None,  # FSQ levels per dimension, e.g., [8, 8, 8]
        fsq_use_projection: bool = True,  # Use projection layer to reduce dim for FSQ
        use_post_quant_conv: bool = False,  # Use post-quantization convolution (Cosmos-Tokenizer style)
        post_quant_conv_kernel_size: int = 1,  # Kernel size for post_quant_conv (1 for 1x1 conv)
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
        
        # Initialize vector quantizer (FSQ, SimVQ, or standard VQ)
        self.num_embeddings = num_embeddings
        self.quantize_before_reshape = quantize_before_reshape
        self.use_simvq = use_simvq
        self.use_fsq = use_fsq
        self.fsq_use_projection = fsq_use_projection
        self.freeze_encoder = freeze_encoder
        self.use_post_quant_conv = use_post_quant_conv
        
        # For tracking losses
        self.last_vq_loss = None
        self.last_commit_loss = None
        self.last_codebook_loss = None
        
        # FSQ projection layer (if needed)
        self.fsq_projection = None
        self.fsq_unprojection = None
        
        # Post-quantization convolution layer (Cosmos-Tokenizer style)
        self.post_quant_conv = None
        
        if use_fsq:
            # Use FSQ (Finite Scalar Quantization)
            from .fsq import FSQ, AttnFSQProjector, FourierFSQUnprojector, FSQAdapter
            
            fsq_dim = len(fsq_levels)
            
            # If latent dimension doesn't match FSQ dimension, use projection
            if fsq_dim != self.latent_dim:
                if not fsq_use_projection:
                    raise ValueError(
                        f"FSQ levels dimension {fsq_dim} does not match "
                        f"latent dimension {self.latent_dim}. "
                        f"Either set fsq_use_projection=True to use a projection layer, "
                        f"or specify fsq_levels with length {self.latent_dim}."
                    )
                
                # Create projection layers to/from FSQ dimension
                # self.fsq_projection = AttnFSQProjector(self.latent_dim, fsq_dim)
                self.fsq_projection = FSQAdapter(self.latent_dim, fsq_dim)
                self.fsq_unprojection = FourierFSQUnprojector(fsq_dim, self.latent_dim)
                print(f"Using projection layer: {self.latent_dim} -> {fsq_dim} -> {self.latent_dim}")
            
            self.quantizer = FSQ(
                levels=fsq_levels,
                eps=vq_epsilon,
            )
            # Calculate effective codebook size
            effective_codebook_size = 1
            for level in fsq_levels:
                effective_codebook_size *= level
            self.num_embeddings = effective_codebook_size
            print(f"Initialized VQRAE with FSQ (levels: {fsq_levels}, "
                  f"implicit codebook size: {effective_codebook_size}, dim: {fsq_dim})")
        elif use_simvq:
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
        
        # Initialize post-quantization convolution if requested (Cosmos-Tokenizer style)
        if use_post_quant_conv:
            # Following Cosmos-Tokenizer: post_quant_conv operates on FSQ codes (before unprojection)
            # For FSQ with projection, we work in fsq_dim space before unprojection
            # For other quantizers or FSQ without projection, we work directly in latent_dim
            padding = post_quant_conv_kernel_size // 2
            if use_fsq and self.fsq_projection is not None:
                # post_quant_conv processes FSQ codes before unprojection to latent_dim
                conv_dim = len(fsq_levels)
                self.post_quant_conv = nn.Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=post_quant_conv_kernel_size,
                    stride=1,
                    padding=padding
                )
                print(f"Initialized post_quant_conv in FSQ space: {conv_dim} -> {conv_dim} "
                      f"(kernel_size={post_quant_conv_kernel_size}, before unprojection to {self.latent_dim})")
            else:
                # Standard case: work in latent_dim space
                self.post_quant_conv = nn.Conv2d(
                    self.latent_dim,
                    self.latent_dim,
                    kernel_size=post_quant_conv_kernel_size,
                    stride=1,
                    padding=padding
                )
                print(f"Initialized post_quant_conv: {self.latent_dim} -> {self.latent_dim} "
                      f"(kernel_size={post_quant_conv_kernel_size})")
        
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
            if self.use_fsq and self.fsq_projection is not None:
                # Project to FSQ dimension
                z_proj = self.fsq_projection(z)
                z_q_proj, vq_loss, indices = self.quantizer(z_proj)
                
                # Apply post_quant_conv in FSQ space BEFORE unprojection (Cosmos-Tokenizer style)
                if self.use_post_quant_conv and self.post_quant_conv is not None:
                    # Reshape to 2D for conv: (B, N, fsq_dim) -> (B, fsq_dim, H, W)
                    b, n, fsq_dim = z_q_proj.shape
                    h = w = int(n ** 0.5)
                    z_q_proj = z_q_proj.transpose(1, 2).view(b, fsq_dim, h, w)
                    z_q_proj = self.post_quant_conv(z_q_proj)
                    # Reshape back: (B, fsq_dim, H, W) -> (B, N, fsq_dim)
                    z_q_proj = z_q_proj.view(b, fsq_dim, n).transpose(1, 2).contiguous()
                
                # Project back to original dimension
                z_q = self.fsq_unprojection(z_q_proj)
            else:
                z_q, vq_loss, indices = self.quantizer(z)
                
                # Apply post_quant_conv if not using FSQ projection
                if self.use_post_quant_conv and self.post_quant_conv is not None:
                    # Reshape to 2D for conv: (B, N, C) -> (B, C, H, W)
                    b, n, c = z_q.shape
                    h = w = int(n ** 0.5)
                    z_q = z_q.transpose(1, 2).view(b, c, h, w)
                    z_q = self.post_quant_conv(z_q)
                    # Reshape back: (B, C, H, W) -> (B, N, C)
                    z_q = z_q.view(b, c, n).transpose(1, 2).contiguous()
            
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
            if self.use_fsq and self.fsq_projection is not None:
                # Need to handle 2D case
                original_shape = z.shape
                b, c, h, w = z.shape
                # Reshape to (B, H, W, C) for projection
                z_reshaped = z.permute(0, 2, 3, 1).contiguous()
                z_proj = self.fsq_projection(z_reshaped)
                # Reshape to (B, fsq_dim, H, W) for quantizer
                z_proj = z_proj.permute(0, 3, 1, 2).contiguous()
                z_q_proj, vq_loss, indices = self.quantizer(z_proj)
                
                # Apply post_quant_conv in FSQ space BEFORE unprojection (Cosmos-Tokenizer style)
                if self.use_post_quant_conv and self.post_quant_conv is not None:
                    # z_q_proj is already in (B, fsq_dim, H, W) format
                    z_q_proj = self.post_quant_conv(z_q_proj)
                
                # Project back: (B, fsq_dim, H, W) -> (B, H, W, fsq_dim)
                z_q_proj = z_q_proj.permute(0, 2, 3, 1).contiguous()
                z_q_reshaped = self.fsq_unprojection(z_q_proj)
                # Reshape back to (B, C, H, W)
                z_q = z_q_reshaped.permute(0, 3, 1, 2).contiguous()
            else:
                z_q, vq_loss, indices = self.quantizer(z)
                
                # Apply post_quant_conv if not using FSQ projection
                if self.use_post_quant_conv and self.post_quant_conv is not None:
                    # z_q is already in (B, C, H, W) format
                    z_q = self.post_quant_conv(z_q)
        
        # Store VQ loss for training (keep gradients for backprop)
        self.last_vq_loss = vq_loss
        # Store commitment loss if available (for SimVQ) - keep gradients
        if hasattr(self.quantizer, 'last_commit_loss'):
            self.last_commit_loss = self.quantizer.last_commit_loss
        
        if hasattr(self.quantizer, 'last_codebook_loss'):
            self.last_codebook_loss = self.quantizer.last_codebook_loss
        
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
            if self.use_fsq and self.fsq_projection is not None:
                # Project to FSQ dimension
                z_proj = self.fsq_projection(z)
                z_q_proj, vq_loss, indices = self.quantizer(z_proj)
                
                # Apply post_quant_conv in FSQ space BEFORE unprojection (Cosmos-Tokenizer style)
                if self.use_post_quant_conv and self.post_quant_conv is not None:
                    # Reshape to 2D for conv: (B, N, fsq_dim) -> (B, fsq_dim, H, W)
                    b, n, fsq_dim = z_q_proj.shape
                    h = w = int(n ** 0.5)
                    z_q_proj = z_q_proj.transpose(1, 2).view(b, fsq_dim, h, w)
                    z_q_proj = self.post_quant_conv(z_q_proj)
                    # Reshape back: (B, fsq_dim, H, W) -> (B, N, fsq_dim)
                    z_q_proj = z_q_proj.view(b, fsq_dim, n).transpose(1, 2).contiguous()
                
                # Project back to original dimension
                z_q = self.fsq_unprojection(z_q_proj)
            else:
                z_q, vq_loss, indices = self.quantizer(z)
                
                # Apply post_quant_conv if not using FSQ projection
                if self.use_post_quant_conv and self.post_quant_conv is not None:
                    # Reshape to 2D for conv: (B, N, C) -> (B, C, H, W)
                    b, n, c = z_q.shape
                    h = w = int(n ** 0.5)
                    z_q = z_q.transpose(1, 2).view(b, c, h, w)
                    z_q = self.post_quant_conv(z_q)
                    # Reshape back: (B, C, H, W) -> (B, N, C)
                    z_q = z_q.view(b, c, n).transpose(1, 2).contiguous()
            
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
            
            if self.use_fsq and self.fsq_projection is not None:
                # Need to handle 2D case
                b, c, h, w = z.shape
                # Reshape to (B, H, W, C) for projection
                z_reshaped = z.permute(0, 2, 3, 1).contiguous()
                z_proj = self.fsq_projection(z_reshaped)
                # Reshape to (B, fsq_dim, H, W) for quantizer
                z_proj = z_proj.permute(0, 3, 1, 2).contiguous()
                z_q_proj, vq_loss, indices = self.quantizer(z_proj)
                
                # Apply post_quant_conv in FSQ space BEFORE unprojection (Cosmos-Tokenizer style)
                if self.use_post_quant_conv and self.post_quant_conv is not None:
                    # z_q_proj is already in (B, fsq_dim, H, W) format
                    z_q_proj = self.post_quant_conv(z_q_proj)
                
                # Project back: (B, fsq_dim, H, W) -> (B, H, W, fsq_dim)
                z_q_proj = z_q_proj.permute(0, 2, 3, 1).contiguous()
                z_q_reshaped = self.fsq_unprojection(z_q_proj)
                # Reshape back to (B, C, H, W)
                z_q = z_q_reshaped.permute(0, 3, 1, 2).contiguous()
            else:
                z_q, vq_loss, indices = self.quantizer(z)
                
                # Apply post_quant_conv if not using FSQ projection
                if self.use_post_quant_conv and self.post_quant_conv is not None:
                    # z_q is already in (B, C, H, W) format
                    z_q = self.post_quant_conv(z_q)
        
        # Store commitment loss if available (for SimVQ) - keep gradients for training
        if hasattr(self.quantizer, 'last_commit_loss'):
            self.last_commit_loss = self.quantizer.last_commit_loss
        
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
        
        Following Cosmos-Tokenizer approach:
        1. Convert indices to codes (codebook lookup) - returns FSQ codes
        2. Apply post_quant_conv on FSQ codes (before unprojection)
        3. Unproject FSQ codes to latent dimension (if using projection)
        4. Apply normalization if enabled
        5. Decode to image
        
        Args:
            indices: Codebook indices
            spatial_shape: Optional (height, width) of latent spatial dimensions.
                          If None, inferred from indices shape.
        
        Returns:
            x_rec: Reconstructed image
        """
        # Step 1: Convert indices to codes (codebook lookup)
        # This returns codes in FSQ space: (B, H, W, fsq_dim)
        z_q_fsq = self.quantizer.get_codebook_entry(indices)
        
        # Step 2: Apply post_quant_conv on FSQ codes BEFORE unprojection (Cosmos-Tokenizer style)
        if self.use_post_quant_conv and self.post_quant_conv is not None:
            # Convert to (B, fsq_dim, H, W) for Conv2d
            z_q_fsq = z_q_fsq.permute(0, 3, 1, 2).contiguous()
            z_q_fsq = self.post_quant_conv(z_q_fsq)
            # Convert back to (B, H, W, fsq_dim)
            z_q_fsq = z_q_fsq.permute(0, 2, 3, 1).contiguous()
        
        # Step 3: Unproject from FSQ space to latent space if needed
        if self.use_fsq and self.fsq_projection is not None:
            # Unproject: (B, H, W, fsq_dim) -> (B, H, W, latent_dim)
            z_q = self.fsq_unprojection(z_q_fsq)
        else:
            z_q = z_q_fsq
        
        # Step 4: Reshape to expected format for decoder
        if self.quantize_before_reshape:
            # z_q is in (B, H, W, C) format
            # Need to convert to (B, N, C) then potentially to (B, C, H, W)
            b, h, w, c = z_q.shape
            z_q = z_q.view(b, h * w, c)
            
            if self.reshape_to_2d:
                z_q = z_q.transpose(1, 2).view(b, c, h, w)
        else:
            # z_q is in (B, H, W, C) format, need to convert to (B, C, H, W)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # Step 5: Apply normalization if enabled
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z_q.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z_q.device) if self.latent_var is not None else 1
            z_q = (z_q - latent_mean) / torch.sqrt(latent_var + self.eps)
        
        # Step 6: Decode to image
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
