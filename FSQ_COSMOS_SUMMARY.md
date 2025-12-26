# FSQ with Cosmos-Tokenizer Implementation

This PR implements FSQuantizer following the NVIDIA Cosmos-Tokenizer approach with a 64K codebook.

## What's New

### 1. FSQ with 64K Codebook
- Implemented FSQ levels [8, 8, 8, 5, 5, 5] for 64,000 codes (~64K)
- Matches the configuration used in NVIDIA Cosmos-Tokenizer

### 2. Post-Quantization Convolution
- Added `post_quant_conv` layer that processes codes after quantization
- Follows Cosmos-Tokenizer architecture: indices → codes → post_quant_conv → decoder
- Configurable kernel size (default: 1x1 convolution)

### 3. Updated VQRAE Class
New parameters:
- `use_post_quant_conv`: Enable post-quantization convolution
- `post_quant_conv_kernel_size`: Kernel size for the convolution layer

## Files Changed

### Core Implementation
- `src/stage1/vqrae.py`: Added post_quant_conv support to VQRAE
  - New initialization parameters
  - Updated `encode()`, `forward()`, and `decode_from_indices()` methods
  - Proper flow: quantization → indices → codes → post_quant_conv → normalization → decoder

### Configuration
- `configs/stage1/training/VQRAE-FSQ-Cosmos-64K.yaml`: Example configuration
  - FSQ levels: [8, 8, 8, 5, 5, 5]
  - Enabled post_quant_conv with 1x1 kernel
  - Uses SigLIP-L encoder with ViT-XL decoder

### Testing
- `test_fsq_cosmos.py`: Comprehensive test suite
  - Verifies 64K codebook size
  - Tests FSQ quantization and indices↔codes conversion
  - Validates post_quant_conv flow

### Documentation
- `docs/FSQ_COSMOS_IMPLEMENTATION.md`: Detailed documentation
  - Architecture overview
  - Implementation details
  - Usage examples
  - Comparison with standard FSQ

## How to Use

### Option 1: Using Configuration File

```bash
python src/train_stage1.py \
  --config configs/stage1/training/VQRAE-FSQ-Cosmos-64K.yaml \
  --data-path /path/to/imagenet \
  --results-dir results/fsq-cosmos
```

### Option 2: Programmatic Usage

```python
from stage1.vqrae import VQRAE

model = VQRAE(
    encoder_cls='SigLIP2wNorm',
    decoder_config_path='configs/decoder/ViTXL',
    use_fsq=True,
    fsq_levels=[8, 8, 8, 5, 5, 5],  # 64K codebook
    use_post_quant_conv=True,        # Enable Cosmos-Tokenizer style
    post_quant_conv_kernel_size=1,   # 1x1 convolution
)

# Encode and decode
indices = model.encode_to_indices(image)
reconstructed = model.decode_from_indices(indices)
```

## Testing

Run the test suite to verify the implementation:

```bash
python test_fsq_cosmos.py
```

Expected output:
```
============================================================
Test 1: FSQ Codebook Size
✓ PASSED: Codebook size is 64000 (64K)

Test 2: FSQ Quantization
✓ PASSED: FSQ quantization and indices<->codes conversion work correctly

Test 4: decode_from_indices Flow
✓ PASSED: indices -> codes -> post_quant_conv flow works correctly

All tests passed!
============================================================
```

## Key Differences from Standard FSQ

| Feature | Standard FSQ | Cosmos-Tokenizer FSQ |
|---------|-------------|---------------------|
| Codebook Size | Configurable | Fixed at 64K (8,8,8,5,5,5) |
| Post-processing | None | Learned post_quant_conv |
| Decode Flow | Quantize → Decode | Quantize → Codes → Conv → Decode |

## Backward Compatibility

The implementation is fully backward compatible:
- Existing FSQ code works unchanged
- `use_post_quant_conv=False` (default) disables the new feature
- Existing configs continue to work without modification

## References

- [NVIDIA Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer)
- [FSQ Paper: Finite Scalar Quantization](https://arxiv.org/abs/2309.15505)
