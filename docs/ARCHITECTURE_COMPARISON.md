# Architecture Comparison: Standard FSQ vs Cosmos-Tokenizer FSQ

## Standard FSQ Flow

```
┌─────────┐      ┌─────────┐      ┌──────────┐      ┌─────────┐
│  Image  │ ---> │ Encoder │ ---> │   FSQ    │ ---> │ Decoder │ ---> │ Reconstructed │
└─────────┘      └─────────┘      │Quantizer │      └─────────┘      │     Image     │
                                   └──────────┘                        └───────────────┘
                                        │
                                        v
                                   [indices]
```

### Processing Steps:
1. Image → Encoder → Continuous latent features
2. Features → FSQ Quantizer → Quantized features + Indices
3. Quantized features → (optional normalization) → Decoder → Reconstructed image

## Cosmos-Tokenizer FSQ Flow (Implemented)

```
┌─────────┐      ┌─────────┐      ┌──────────┐      ┌─────────┐      ┌──────────────┐      ┌─────────┐
│  Image  │ ---> │ Encoder │ ---> │   FSQ    │ ---> │ indices │ ---> │indices_to    │ ---> │post_    │
└─────────┘      └─────────┘      │Quantizer │      │         │      │_codes()      │      │quant_   │
                                   └──────────┘      └─────────┘      │ (codebook    │      │conv     │
                                                                       │  lookup)     │      │         │
                                                                       └──────────────┘      └─────────┘
                                                                                                  │
                                                                                                  v
                                                                                             [processed
                                                                                               codes]
                                                                                                  │
                                                                                                  v
                                                                       ┌──────────────┐      ┌─────────┐
                                                                       │ Normalization│ <--- │         │
                                                                       │  (optional)  │      │         │
                                                                       └──────────────┘      └─────────┘
                                                                                                  │
                                                                                                  v
                                                                       ┌──────────────┐      ┌─────────┐
                                                                       │   Decoder    │ ---> │Reconstruct│
                                                                       │              │      │   Image   │
                                                                       └──────────────┘      └───────────┘
```

### Processing Steps:
1. Image → Encoder → Continuous latent features
2. Features → FSQ Quantizer → Quantized features + **Indices**
3. **Indices → Codebook Lookup (indices_to_codes) → Pure quantized codes**
4. **Codes → post_quant_conv (Conv2d) → Processed codes** ← **NEW**
5. Processed codes → (optional normalization) → Decoder → Reconstructed image

## Key Architectural Differences

### 1. Quantization Output
- **Standard FSQ**: Returns quantized features with straight-through gradient
- **Cosmos FSQ**: Uses indices as intermediate representation

### 2. Post-Quantization Processing
- **Standard FSQ**: Direct path from quantization to decoder
- **Cosmos FSQ**: Learned post_quant_conv transforms codes before decoding

### 3. Decoding from Indices
```python
# Standard FSQ
def decode(z_q):
    return decoder(z_q)

# Cosmos-Tokenizer FSQ
def decode_from_indices(indices):
    codes = quantizer.indices_to_codes(indices)  # Step 1: Codebook lookup
    codes = post_quant_conv(codes)                # Step 2: Learned transformation
    codes = normalize(codes)                       # Step 3: Optional normalization
    return decoder(codes)                          # Step 4: Decode to image
```

## FSQ Codebook Details

### Standard FSQ Configuration
```python
fsq_levels = [8, 8, 8]  # 512 codes
# or
fsq_levels = [8, 5, 5, 5]  # 1000 codes
```

### Cosmos-Tokenizer Configuration (Implemented)
```python
fsq_levels = [8, 8, 8, 5, 5, 5]  # 64,000 codes (~64K)
```

**Calculation**: 8 × 8 × 8 × 5 × 5 × 5 = 64,000

### Quantization Levels
For each dimension with L levels:
- Values: [-(L-1)/2, ..., -1/2, 1/2, ..., (L-1)/2]

Example for level 8:
```
[-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
```

Example for level 5:
```
[-2.0, -1.0, 0.0, 1.0, 2.0]
```

## post_quant_conv Layer

### Architecture
```python
post_quant_conv = nn.Conv2d(
    in_channels=latent_dim,    # e.g., 768 for SigLIP-L
    out_channels=latent_dim,   # Same as input
    kernel_size=1,             # 1x1 convolution (default)
    stride=1,
    padding=0
)
```

### Purpose
- Learn optimal transformation of discrete codes
- Can adjust code statistics before decoding
- Potentially improves reconstruction quality
- Adds minimal computational overhead (1×1 conv)

### Input/Output
- **Input**: Quantized codes in (B, C, H, W) format
- **Output**: Transformed codes in (B, C, H, W) format
- No change in spatial or channel dimensions

## Implementation Flexibility

### Configuration Options

```yaml
# Option 1: Cosmos-Tokenizer style (NEW)
use_fsq: true
fsq_levels: [8, 8, 8, 5, 5, 5]
use_post_quant_conv: true
post_quant_conv_kernel_size: 1

# Option 2: Standard FSQ
use_fsq: true
fsq_levels: [8, 8, 8]
use_post_quant_conv: false

# Option 3: Standard VQ-VAE
use_fsq: false
num_embeddings: 16384
```

### Code Example

```python
# Enable Cosmos-Tokenizer features
model = VQRAE(
    use_fsq=True,
    fsq_levels=[8, 8, 8, 5, 5, 5],
    use_post_quant_conv=True,
)

# Standard FSQ (backward compatible)
model = VQRAE(
    use_fsq=True,
    fsq_levels=[8, 8, 8],
    use_post_quant_conv=False,  # or omit (default)
)
```

## Performance Considerations

### Computational Overhead
- **post_quant_conv (1×1)**: Minimal overhead (~0.01% increase)
- **Codebook size**: 64K vs 512 → No runtime difference (implicit codebook)
- **Memory**: Slightly increased for storing post_quant_conv weights

### Expected Benefits
1. **Larger codebook** (64K) → Better representational capacity
2. **Learned post-processing** → Potentially improved reconstruction
3. **Flexible architecture** → Can be tuned for specific tasks

## Backward Compatibility

✅ All existing code continues to work
✅ New features are opt-in via configuration
✅ No breaking changes to API
✅ Tests validate both old and new flows
