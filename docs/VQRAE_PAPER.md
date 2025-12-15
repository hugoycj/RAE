# VQRAE Implementation Based on Paper Specifications

## Overview

This implementation of VQRAE follows the specifications from the research paper, implementing a two-stage training approach with SimVQ (Similarity-based Vector Quantization).

## Key Specifications (From Paper)

### Architecture

| Component | Specification | Details |
|-----------|--------------|---------|
| **Encoder** | SigLIP-L (400M) / InternViT-6B | Pretrained vision foundation model |
| **Patch Size** | 16 | Standard ViT patch size |
| **Input Resolution** | 224×224 (training) / Variable (inference) | Multi-resolution support |
| **Codebook Dimension** | 1536 | Matches encoder output (high-dim semantic quantization) |
| **Codebook Size** | 16384 (16k) | Optimal configuration, 100% utilization |
| **VQ Method** | SimVQ | Learnable projection matrix, avoids collapse |
| **Decoder** | Symmetric ViT | Mirrors encoder (24 layers for SigLIP-L) |

### Training

| Aspect | Specification | Details |
|--------|--------------|---------|
| **Training Stages** | Two-stage | Stage-1: Frozen encoder, Stage-2: Unfrozen + distillation |
| **Stage-1 Losses** | L₂ + LPIPS + GAN | GAN is patch-based discriminator |
| **Stage-2 Loss** | Self-distillation | L₂ on continuous → quantized features |
| **Dataset** | DataComp-1B | 1.1B images for fair comparison |
| **Batch Size** | 4096 (global) | 512 per GPU × 8 GPUs |
| **Epochs** | 30 | Full training duration |
| **LR (Stage-1)** | 1e-3 | For codebook & decoder |
| **LR (Stage-2)** | 1e-5 | For full model fine-tuning |

## New Features

### 1. SimVQ (Similarity-based Vector Quantization)

Located in `src/stage1/simvq.py`:

```python
from stage1.simvq import SimVQ

quantizer = SimVQ(
    num_embeddings=16384,
    embedding_dim=1536,
    use_projection=True,  # Key feature
)
```

**Key differences from standard VQ:**
- Uses cosine similarity instead of L2 distance
- Learnable projection matrices prevent codebook collapse
- More stable training
- Better semantic preservation

### 2. Two-Stage Training

**Stage-1: Frozen Encoder** (`configs/stage1/training/VQRAE-SigLIP-L_Stage1.yaml`)
- Encoder frozen (no gradients)
- Train codebook and decoder
- Losses: L2 + LPIPS + GAN
- LR: 1e-3

**Stage-2: Unfrozen Encoder + Distillation** (`configs/stage1/training/VQRAE-SigLIP-L_Stage2.yaml`)
- Encoder unfrozen (gradients flow)
- Self-distillation loss
- Losses: L2 + LPIPS (reduced) + Distillation
- LR: 1e-5 (much lower)
- No GAN loss

### 3. Encoder Flexibility

Now supports multiple encoders:
- **SigLIP-L**: Primary (400M params, 1536 dim)
- **InternViT-6B**: SOTA (6B params)
- **DINOv2-B**: Baseline (86M params, 768 dim)

### 4. Codebook Utilization Monitoring

```python
model = VQRAE(use_simvq=True)
usage = model.get_codebook_usage()
print(f"Codebook utilization: {usage*100:.1f}%")  # Should be ~100%
```

## Usage

### Training Stage-1

```bash
torchrun --standalone --nproc_per_node=8 \
  src/train_stage1.py \
  --config configs/stage1/training/VQRAE-SigLIP-L_Stage1.yaml \
  --data-path /path/to/datacomp1b \
  --results-dir results/vqrae_stage1 \
  --precision bf16
```

### Training Stage-2

```bash
torchrun --standalone --nproc_per_node=8 \
  src/train_stage2.py \
  --config configs/stage1/training/VQRAE-SigLIP-L_Stage2.yaml \
  --resume results/vqrae_stage1/checkpoint_final.pt \
  --data-path /path/to/datacomp1b \
  --results-dir results/vqrae_stage2 \
  --precision bf16
```

### Inference

```python
from stage1 import VQRAE
import torch

# Load trained model
model = VQRAE(
    encoder_cls='SigLIP2wNorm',
    encoder_config_path='google/siglip-large-patch16-384',
    num_embeddings=16384,
    use_simvq=True,
    freeze_encoder=False,  # For inference
)
model.load_state_dict(torch.load('vqrae_stage2_final.pt'))
model.eval()

# Encode to discrete codes
with torch.no_grad():
    indices = model.encode_to_indices(image)
    print(f"Image encoded to {indices.shape} discrete codes")

# Reconstruct from codes
with torch.no_grad():
    reconstructed = model.decode_from_indices(indices)
```

## Configuration Files

### Stage-1 Config
`configs/stage1/training/VQRAE-SigLIP-L_Stage1.yaml`:
- SigLIP-L encoder (frozen)
- 16k codebook with SimVQ
- L2 + LPIPS + GAN losses
- Batch size: 4096, LR: 1e-3, 30 epochs

### Stage-2 Config
`configs/stage1/training/VQRAE-SigLIP-L_Stage2.yaml`:
- SigLIP-L encoder (unfrozen)
- Continue from Stage-1 checkpoint
- Self-distillation loss
- Batch size: 4096, LR: 1e-5, 10 epochs

## Key Improvements Over Initial Implementation

1. **SimVQ**: Replaces standard VQ for better stability
2. **Two-Stage Training**: Preserves semantic understanding
3. **Larger Codebook**: 16k instead of 8k for better capacity
4. **Higher Dimensions**: 1536 instead of 768 (SigLIP-L)
5. **Distillation Loss**: Maintains frozen encoder knowledge
6. **Codebook Monitoring**: Track utilization rate

## Distillation Loss (Stage-2)

The self-distillation loss is crucial for Stage-2:

```python
# During forward pass
continuous_features = encoder(x)  # Before quantization
quantized_features = quantizer(continuous_features)

# Distillation loss
distill_loss = F.mse_loss(quantized_features, continuous_features.detach())

# Total loss (Stage-2)
total_loss = recon_loss + lpips_loss + vq_loss + distill_weight * distill_loss
```

This ensures that even when the encoder is unfrozen, the quantized features remain semantically meaningful.

## Performance Tips

1. **Use bf16**: Significantly faster training
2. **Gradient Accumulation**: If GPU memory is limited
3. **Monitor Codebook Usage**: Should approach 100%
4. **Stage-1 First**: Always complete Stage-1 before Stage-2
5. **Lower LR in Stage-2**: Critical for stability

## Expected Results

With proper training:
- **Codebook Utilization**: ~100% (all 16k entries used)
- **Reconstruction Quality**: High fidelity with LPIPS < 0.1
- **Semantic Preservation**: Minimal drop in downstream task performance
- **Compression**: ~8-16x depending on configuration

## Differences from Initial VQRAE

| Feature | Initial | Paper-based |
|---------|---------|-------------|
| VQ Method | Standard EMA | SimVQ with projection |
| Codebook Size | 8192 | 16384 |
| Encoder | DINOv2-B | SigLIP-L / InternViT-6B |
| Training | Single-stage | Two-stage |
| Distillation | No | Yes (Stage-2) |
| Codebook Dim | 768 | 1536 |

## References

- SimVQ: Similarity-based Vector Quantization
- SigLIP: Sigmoid Loss for Language Image Pre-Training
- DataComp: A large-scale data filtering benchmark
