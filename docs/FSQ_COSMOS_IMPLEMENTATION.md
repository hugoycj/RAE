# FSQuantizer Implementation following NVIDIA Cosmos-Tokenizer

## Overview

This implementation follows the NVIDIA Cosmos-Tokenizer approach for Finite Scalar Quantization (FSQ) with a 64K codebook. The key architectural change is the addition of a `post_quant_conv` layer that processes quantized codes before they are fed into the decoder.

## Architecture Flow

```
Encoder → Quantization → Indices → Codes → post_quant_conv → Normalization → Decoder
```

### Cosmos-Tokenizer Style Processing

1. **Encoding & Quantization**: Input image → Encoder → Continuous features → FSQ Quantization → Indices
2. **Indices to Codes**: Indices → Codebook lookup → Quantized codes
3. **Post-Quantization Convolution**: Codes → post_quant_conv (Conv2d) → Processed codes
4. **Decoding**: Processed codes → (optional normalization) → Decoder → Reconstructed image

## Key Features

### 1. FSQ with 64K Codebook

- **FSQ Levels**: [8, 8, 8, 5, 5, 5]
- **Codebook Size**: 8 × 8 × 8 × 5 × 5 × 5 = **64,000 codes** (~64K)
- **Dimension**: 6-dimensional quantization space

### 2. Post-Quantization Convolution

The `post_quant_conv` is a learned convolutional layer that:
- Takes quantized codes in (B, C, H, W) format
- Applies a convolution (default: 1×1) to process the codes
- Outputs processed codes in the same shape
- Acts as a learned transformation between quantization and decoding

**Purpose**: Allows the model to learn an optimal transformation of the discrete codes before reconstruction, potentially improving image quality.

### 3. Projection Layers (Optional)

When the FSQ dimension (6) doesn't match the encoder's latent dimension (e.g., 768):
- **fsq_projection**: Linear layer that projects from latent_dim → fsq_dim
- **fsq_unprojection**: Linear layer that projects back fsq_dim → latent_dim

This allows using low-dimensional FSQ with high-dimensional encoders.

## Implementation Details

### VQRAE Parameters

```python
VQRAE(
    # FSQ parameters
    use_fsq=True,
    fsq_levels=[8, 8, 8, 5, 5, 5],  # 64K codebook
    fsq_use_projection=True,          # Use projection layers
    
    # Cosmos-Tokenizer style parameters
    use_post_quant_conv=True,         # Enable post-quant conv
    post_quant_conv_kernel_size=1,    # 1x1 convolution
    
    # Other parameters...
)
```

### decode_from_indices Method

The key method that follows the Cosmos-Tokenizer approach:

```python
def decode_from_indices(self, indices):
    # Step 1: Convert indices to codes (codebook lookup)
    z_q = self.quantizer.get_codebook_entry(indices)
    
    # Step 2: Reshape to (B, C, H, W) format
    z_q = reshape_codes(z_q)
    
    # Step 3: Apply post_quant_conv (Cosmos-Tokenizer style)
    if self.use_post_quant_conv:
        z_q = self.post_quant_conv(z_q)
    
    # Step 4: Apply normalization if enabled
    if self.do_normalization:
        z_q = normalize(z_q)
    
    # Step 5: Decode to image
    return self.decode(z_q)
```

## Configuration Example

See `configs/stage1/training/VQRAE-FSQ-Cosmos-64K.yaml`:

```yaml
stage_1:
  target: stage1.VQRAE
  params:
    # FSQ configuration
    use_fsq: true
    fsq_levels: [8, 8, 8, 5, 5, 5]  # 64K codebook
    fsq_use_projection: true
    
    # Cosmos-Tokenizer style post-quantization convolution
    use_post_quant_conv: true
    post_quant_conv_kernel_size: 1
    
    # Encoder/Decoder configuration
    encoder_cls: 'SigLIP2wNorm'
    decoder_config_path: 'configs/decoder/ViTXL'
    # ... other params
```

## Testing

Run the test suite to verify the implementation:

```bash
python test_fsq_cosmos.py
```

### Test Coverage

1. **Codebook Size**: Verifies FSQ produces exactly 64,000 codes
2. **Quantization**: Tests FSQ forward pass and indices↔codes conversion
3. **Post-Quant Conv Flow**: Validates the indices → codes → post_quant_conv pipeline

## Comparison with Standard FSQ

| Feature | Standard FSQ | Cosmos-Tokenizer FSQ |
|---------|-------------|---------------------|
| Quantization | ✓ | ✓ |
| Codebook Size | Configurable | 64K (8,8,8,5,5,5) |
| Post-quant Conv | ✗ | ✓ |
| Processing Flow | Quantize → Decode | Quantize → Codes → Conv → Decode |

## Benefits

1. **Larger Codebook**: 64K codes provide more representational capacity
2. **Learned Post-Processing**: post_quant_conv can learn optimal transformations
3. **Flexible Integration**: Can be enabled/disabled via configuration
4. **Backward Compatible**: Existing FSQ code still works when post_quant_conv is disabled

## Usage Example

```python
from stage1.vqrae import VQRAE

# Create VQRAE with Cosmos-Tokenizer style FSQ
model = VQRAE(
    encoder_cls='SigLIP2wNorm',
    decoder_config_path='vit_mae-base',
    use_fsq=True,
    fsq_levels=[8, 8, 8, 5, 5, 5],
    use_post_quant_conv=True,
    post_quant_conv_kernel_size=1,
)

# Encode image to indices
indices = model.encode_to_indices(image)

# Decode from indices (with post_quant_conv)
reconstructed = model.decode_from_indices(indices)
```

## References

- NVIDIA Cosmos-Tokenizer: https://github.com/NVIDIA/Cosmos-Tokenizer
- FSQ Paper: "Finite Scalar Quantization: VQ-VAE Made Simple" (arXiv:2309.15505)
