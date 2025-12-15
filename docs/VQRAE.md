# VQRAE: Vector Quantized Representation Autoencoder

This directory contains the implementation of VQRAE, which extends the standard RAE (Representation Autoencoder) with vector quantization to enable discrete latent representations.

## Overview

VQRAE combines:
- **Pretrained frozen encoders** (DINOv2, SigLIP, MAE) from RAE
- **Vector Quantization** for discrete latent codes
- **Learned ViT decoders** for reconstruction

The vector quantization layer learns a discrete codebook and maps continuous encoder outputs to discrete codes, enabling:
- More interpretable latent representations
- Better controllability for generation
- Potential for tokenized image representations
- Compatibility with discrete diffusion models

## Architecture

```
Input Image
    ↓
Frozen Encoder (DINOv2/SigLIP/MAE)
    ↓
Continuous Latent z
    ↓
Vector Quantizer → Discrete Codes (indices)
    ↓
Quantized Latent z_q
    ↓
Learned ViT Decoder
    ↓
Reconstructed Image
```

## Files

- `src/stage1/vector_quantizer.py`: Implements the `VectorQuantizer` module with EMA updates
- `src/stage1/vqrae.py`: Implements the `VQRAE` model class extending RAE
- `configs/stage1/pretrained/VQRAE-DINOv2-B.yaml`: Config for pretrained VQRAE
- `configs/stage1/training/VQRAE-DINOv2-B_decXL.yaml`: Training config for VQRAE

## Usage

### Basic Usage

```python
from stage1 import VQRAE

# Initialize VQRAE
model = VQRAE(
    encoder_cls='Dinov2withNorm',
    encoder_config_path='facebook/dinov2-base',
    num_embeddings=8192,  # Codebook size
    commitment_cost=0.25,
    vq_decay=0.99,
)

# Forward pass (reconstruction)
x_rec = model(x)

# Get reconstruction with losses
x_rec, losses = model(x, return_loss=True)
print(f"Recon loss: {losses['recon_loss']}")
print(f"VQ loss: {losses['vq_loss']}")

# Encode to discrete indices
indices = model.encode_to_indices(x)

# Decode from indices
x_rec = model.decode_from_indices(indices)
```

### Training

Train VQRAE using the provided training script:

```bash
torchrun --standalone --nproc_per_node=N \
  src/train_stage1.py \
  --config configs/stage1/training/VQRAE-DINOv2-B_decXL.yaml \
  --data-path /path/to/imagenet \
  --results-dir results/vqrae \
  --image-size 256 \
  --precision bf16
```

### Sampling/Reconstruction

Reconstruct images using a trained VQRAE:

```bash
python src/stage1_sample.py \
  --config configs/stage1/pretrained/VQRAE-DINOv2-B.yaml \
  --image assets/pixabay_cat.png
```

## Configuration Parameters

### Vector Quantization Parameters

- `num_embeddings`: Size of the codebook (default: 8192)
  - Common values: 512, 1024, 8192, 16384
  - Larger codebooks can capture more detail but are harder to train
  
- `commitment_cost`: Weight for the commitment loss (default: 0.25)
  - Encourages encoder outputs to commit to codebook entries
  - Typical range: 0.1 - 0.5
  
- `vq_decay`: Decay rate for EMA codebook updates (default: 0.99)
  - Higher values = more stable but slower adaptation
  - Typical range: 0.95 - 0.999
  
- `vq_epsilon`: Numerical stability constant (default: 1e-5)

- `quantize_before_reshape`: Whether to quantize before reshaping to 2D (default: False)
  - False: Quantize in (B, C, H, W) format (recommended)
  - True: Quantize in (B, N, C) format

### Inherited RAE Parameters

All RAE parameters are supported:
- `encoder_cls`, `encoder_config_path`, `encoder_params`
- `decoder_config_path`, `decoder_patch_size`
- `noise_tau`, `reshape_to_2d`, `normalization_stat_path`

## Vector Quantization Details

The `VectorQuantizer` module implements the VQ mechanism from "Neural Discrete Representation Learning" (van den Oord et al., 2017):

1. **Forward Pass**:
   - Compute distances between encoder outputs and codebook entries
   - Select nearest codebook entry for each spatial location
   - Use straight-through estimator for gradients

2. **Loss Components**:
   - Codebook loss: `||sg[z] - e||^2` (moves codebook toward encoder outputs)
   - Commitment loss: `β||z - sg[e]||^2` (encourages encoder to commit)
   - Where `sg[]` is stop-gradient, `z` is encoder output, `e` is codebook entry

3. **Codebook Updates**:
   - Uses exponential moving average (EMA) for stable training
   - Avoids codebook collapse with Laplace smoothing

## Differences from Standard RAE

| Feature | RAE | VQRAE |
|---------|-----|-------|
| Latent Space | Continuous | Discrete |
| Additional Loss | - | VQ loss (commitment + codebook) |
| Codebook | No | Yes (learned) |
| Gradient Flow | Direct | Straight-through estimator |
| Use Cases | Dense prediction, generation | Tokenization, discrete diffusion |

## Tips for Training

1. **Start with pretrained RAE decoder**: Initialize VQRAE with a pretrained RAE decoder for faster convergence
2. **Tune codebook size**: Larger codebooks (8192+) work better for high-resolution images
3. **Disable noise**: Set `noise_tau=0` as VQ provides regularization
4. **Monitor codebook usage**: Track how many codebook entries are being used
5. **Balance VQ weight**: The `vq_weight` in training config controls the importance of VQ loss

## Performance Considerations

- VQ adds minimal computational overhead (~5-10% vs RAE)
- Memory scales with codebook size (negligible for sizes < 100k)
- EMA updates are efficient and don't require backprop through codebook
- Supports both 2D (B, C, H, W) and 1D (B, N, C) latent formats

## Future Extensions

Possible enhancements:
- Multi-scale VQ with hierarchical codebooks
- Factorized/product VQ for larger effective codebook sizes
- Gumbel-softmax for differentiable discrete sampling
- Codebook reset strategies to prevent collapse
- Integration with discrete diffusion models

## References

- VQ-VAE: "Neural Discrete Representation Learning" (van den Oord et al., 2017)
- VQ-VAE-2: "Generating Diverse High-Fidelity Images with VQ-VAE-2" (Razavi et al., 2019)
- RAE: "Diffusion Transformers with Representation Autoencoders" (Zheng et al., 2024)
