# FSQ (Finite Scalar Quantization) for RAE

This directory contains an implementation of Finite Scalar Quantization (FSQ) as described in the paper ["Finite Scalar Quantization: VQ-VAE Made Simple"](https://arxiv.org/pdf/2309.15505).

## What is FSQ?

FSQ is a simple alternative to traditional Vector Quantization (VQ) that eliminates the learned codebook. Instead, FSQ quantizes each dimension independently to a fixed set of levels, creating an implicit codebook through the Cartesian product of all level sets.

### Key Advantages

- **No codebook learning required**: The codebook is implicitly defined, eliminating codebook collapse
- **100% codebook utilization**: All codes are valid by construction
- **Deterministic and simple**: No EMA updates or complex optimization
- **Fast training and inference**: Simpler computation than VQ-VAE

## Usage

### Basic Usage in VQRAE

To use FSQ in your VQRAE configuration, set the following parameters:

```yaml
stage_1:
  target: stage1.VQRAE
  params:
    # Enable FSQ
    use_fsq: true
    
    # Define quantization levels per dimension
    # The codebook size is the product of all levels
    fsq_levels: [8, 8, 8]  # 8^3 = 512 codes
    
    # Small epsilon for numerical stability
    vq_epsilon: 1e-3
    
    # Other parameters...
    quantize_before_reshape: false
    freeze_encoder: true
```

### FSQ Level Configurations

The `fsq_levels` parameter defines how many quantization levels each dimension has. The total codebook size is the product of all levels.

Common configurations from the paper:

| Configuration | Codebook Size | Description |
|--------------|---------------|-------------|
| `[8, 8, 8]` | 512 | Small codebook, 3D latent |
| `[8, 6, 5]` | 240 | Asymmetric, 3D latent |
| `[7, 5, 5, 5]` | 875 | Medium codebook, 4D latent |
| `[8, 5, 5, 5]` | 1000 | Large codebook, 4D latent |

**Important**: The length of `fsq_levels` must match your encoder's latent dimension. For example, if your encoder outputs 3-dimensional latents, use 3 levels like `[8, 8, 8]`.

### Example Configuration

See `configs/stage1/training/VQRAE-FSQ-SigLIP-L.yaml` for a complete example configuration.

## Direct Usage

You can also use FSQ directly in your code:

```python
from stage1.fsq import FSQ
import torch

# Create FSQ quantizer with 512 codes (8^3)
fsq = FSQ(levels=[8, 8, 8])

# Quantize input tensor
z = torch.randn(4, 3, 16, 16)  # (B, C, H, W)
z_q, loss, indices = fsq(z)

# z_q: quantized output (same shape as input)
# loss: commitment loss (scalar)
# indices: codebook indices (B, H, W)
```

## How FSQ Works

1. **Per-dimension quantization**: Each dimension is independently quantized to one of L levels
2. **Level definition**: For L levels, FSQ uses values: -(L-1)/2, ..., -1/2, 1/2, ..., (L-1)/2
3. **Implicit codebook**: The codebook is the Cartesian product of all level sets
4. **Straight-through estimator**: Gradients flow through using the straight-through trick

### Example

For levels `[8, 6, 5]`:
- Dimension 0: quantized to {-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5}
- Dimension 1: quantized to {-2.5, -1.5, -0.5, 0.5, 1.5, 2.5}
- Dimension 2: quantized to {-2, -1, 0, 1, 2}
- Total codes: 8 × 6 × 5 = 240

## Comparison with VQ and SimVQ

| Feature | VQ | SimVQ | FSQ |
|---------|----|----|-----|
| Codebook learning | Yes (EMA) | Projection only | No |
| Codebook collapse | Possible | Unlikely | Impossible |
| Codebook utilization | Variable | ~100% | 100% |
| Training speed | Medium | Fast | Fastest |
| Memory | High (codebook) | Medium (projection) | Low (implicit) |

## Testing

Run the test suite to verify FSQ implementation:

```bash
# Standalone tests (no dependencies)
python test_fsq_standalone.py

# Integration tests
python test_fsq_integration.py
```

## References

- Paper: [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/pdf/2309.15505)
- Authors: Fabian Mentzer, David Minnen, Eirikur Agustsson, Michael Tschannen
- Published: arXiv:2309.15505 (2023)

## Implementation Details

The FSQ implementation in `src/stage1/fsq.py` includes:

- `FSQ.__init__()`: Initialize with level configuration
- `FSQ.forward()`: Quantize input and compute loss
- `FSQ.get_codebook_entry()`: Retrieve codes from indices
- `FSQ.codes_to_indices()`: Convert codes to indices
- `FSQ.indices_to_codes()`: Convert indices to codes

The implementation is compatible with the existing VQRAE interface and can be used as a drop-in replacement for VQ or SimVQ.
