# SimVQ Implementation Verification

**Date:** 2025-12-18  
**Reference:** https://github.com/youngsheen/SimVQ  
**Paper:** SimVQ: Addressing Representation Collapse in Vector Quantized Models with One Linear Layer (arXiv:2411.02038)

---

## Summary

The SimVQ implementation has been **corrected** to match the reference implementation. The key issue was using cosine similarity instead of L2 distance.

---

## Corrected Implementation

### Key Components

1. **Frozen Codebook**
   ```python
   self.embedding = nn.Embedding(num_embeddings, embedding_dim)
   nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim**-0.5)
   for p in self.embedding.parameters():
       p.requires_grad = False  # Frozen!
   ```

2. **Learnable Projection Layer**
   ```python
   self.embedding_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
   ```

3. **L2 Distance Computation** (NOT cosine similarity)
   ```python
   quant_codebook = self.embedding_proj(self.embedding.weight)
   distances = (
       torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
       torch.sum(quant_codebook ** 2, dim=1) -
       2 * torch.matmul(z_flattened, quant_codebook.t())
   )
   encoding_indices = torch.argmin(distances, dim=1)  # Minimum distance
   ```

4. **Loss Formulation**
   ```python
   # Legacy (default, for backwards compatibility with reference)
   vq_loss = (
       torch.mean((quantized.detach() - z) ** 2) +
       self.commitment_cost * torch.mean((quantized - z.detach()) ** 2)
   )
   
   # Standard (corrected version)
   vq_loss = (
       self.commitment_cost * torch.mean((quantized.detach() - z) ** 2) +
       torch.mean((quantized - z.detach()) ** 2)
   )
   ```

---

## Changes Made

### Before (Incorrect)

| Aspect | Old Implementation |
|--------|-------------------|
| **Distance metric** | Cosine similarity |
| **Projections** | Two projections (input + codebook) |
| **Normalization** | L2 normalization for cosine |
| **Codebook** | Learnable buffer |
| **Method** | `argmax(similarity)` |

### After (Correct)

| Aspect | New Implementation |
|--------|-------------------|
| **Distance metric** | L2 distance |
| **Projections** | Single projection (codebook only) |
| **Normalization** | None (raw L2 distance) |
| **Codebook** | Frozen nn.Embedding |
| **Method** | `argmin(distance)` |

---

## Verification Tests

### Test 1: Frozen Codebook ✅
```
Codebook frozen: True (requires_grad=False)
Projection learnable: True (requires_grad=True)
```

### Test 2: Forward Pass ✅
```
Input:   torch.Size([2, 512, 14, 14])
Output:  torch.Size([2, 512, 14, 14])
Indices: torch.Size([2, 14, 14])
Loss:    1.2446
```

### Test 3: Gradient Flow ✅
```
✓ Input receives gradients
✓ Projection layer receives gradients
✓ Codebook does NOT receive gradients (frozen)
```

### Test 4: L2 Distance Computation ✅
```
Manual computation matches forward pass
Uses L2 distance (not cosine similarity)
```

---

## Comparison with Reference

### Code Structure

**Reference (youngsheen/SimVQ):**
```python
# From taming/modules/vqvae/quantize.py
self.embedding = nn.Embedding(self.n_e, self.e_dim)
nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)
for p in self.embedding.parameters():
    p.requires_grad = False

self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)

quant_codebook = self.embedding_proj(self.embedding.weight)
d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
    torch.sum(quant_codebook**2, dim=1) - 2 * \
    torch.einsum('bd,dn->bn', z_flattened, rearrange(quant_codebook, 'n d -> d n'))
```

**Our Implementation:**
```python
# From src/stage1/simvq.py
self.embedding = nn.Embedding(num_embeddings, embedding_dim)
nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim**-0.5)
for p in self.embedding.parameters():
    p.requires_grad = False

self.embedding_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)

quant_codebook = self.embedding_proj(self.embedding.weight)
distances = (
    torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
    torch.sum(quant_codebook ** 2, dim=1) -
    2 * torch.matmul(z_flattened, quant_codebook.t())
)
```

✅ **Matches reference implementation**

---

## Why SimVQ Works

From the paper (arXiv:2411.02038):

> "We find that the key to preventing codebook collapse is to keep the codebook frozen 
> and learn a single linear projection layer. This simple modification achieves 100% 
> codebook utilization."

### The Algorithm:

1. Initialize codebook C with random values
2. Freeze codebook (no gradient updates)
3. Learn projection matrix W
4. Compute distances in projected space: d(z, W·c_i)
5. Select nearest code: argmin_i d(z, W·c_i)
6. Quantize: z_q = W·c_i

### Key Insight:

Instead of updating the codebook to match encoder outputs (which causes collapse), 
SimVQ learns to project the fixed codebook into the right space. The projection layer 
adapts while the codebook provides a stable reference.

---

## Performance Characteristics

Based on reference results:

| Codebook Size | Utilization | rFID | LPIPS |
|--------------|-------------|------|-------|
| 1,024 | 100.0% | 3.67 | 0.16 |
| 8,192 | 100.0% | 2.98 | 0.14 |
| 65,536 | 100.0% | 2.24 | 0.12 |
| 262,144 | 100.0% | 1.99 | 0.11 |

Key advantage: **100% codebook utilization** vs 1-5% for standard VQ-GAN

---

## API Compatibility

The corrected implementation maintains full API compatibility:

```python
from stage1.simvq import SimVQ

# Initialize
simvq = SimVQ(
    num_embeddings=16384,
    embedding_dim=768,
    commitment_cost=0.25,
    legacy=True  # Use legacy loss formulation (default)
)

# Forward pass
z = torch.randn(B, C, H, W)
z_q, vq_loss, indices = simvq(z)

# Codebook entry retrieval
entries = simvq.get_codebook_entry(indices)

# Check usage
usage = simvq.get_codebook_usage()
```

---

## Integration with VQRAE

The VQRAE class automatically uses the corrected SimVQ when `use_simvq=True`:

```python
from stage1 import VQRAE

model = VQRAE(
    encoder_cls='SigLIP2wNorm',
    num_embeddings=16384,
    use_simvq=True,  # Uses corrected SimVQ
    freeze_encoder=True
)
```

---

## References

1. **SimVQ Paper:** arXiv:2411.02038
2. **Reference Code:** https://github.com/youngsheen/SimVQ
3. **Algorithm:** Lines 28-33 in `taming/modules/vqvae/quantize.py`

---

**Verification Status:** ✅ PASSED  
**Match with Reference:** ✅ CONFIRMED  
**Ready for Use:** ✅ YES
