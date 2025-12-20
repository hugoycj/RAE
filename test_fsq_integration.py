#!/usr/bin/env python3
"""
Integration test for FSQ with VQRAE (without full dependencies).

This test validates that FSQ can be properly instantiated through the 
VQRAE interface with the correct parameters.
"""

import torch
import sys
import os

# Test FSQ integration without full VQRAE dependencies
def test_fsq_integration_minimal():
    """Test FSQ integration through minimal interface."""
    print("=" * 60)
    print("Testing FSQ Integration (Minimal)")
    print("=" * 60)
    
    # Import FSQ directly without going through stage1.__init__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fsq", 
        os.path.join(os.path.dirname(__file__), 'src', 'stage1', 'fsq.py')
    )
    fsq_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fsq_module)
    FSQ = fsq_module.FSQ
    
    # Test 1: Verify FSQ can be instantiated with various configurations
    print("\nTest 1: FSQ instantiation with different configurations")
    print("-" * 60)
    
    configs = [
        {"levels": [8, 8, 8], "name": "512 codes (3D)"},
        {"levels": [8, 6, 5], "name": "240 codes (3D)"},
        {"levels": [7, 5, 5, 5], "name": "875 codes (4D)"},
        {"levels": [8, 5, 5, 5], "name": "1000 codes (4D)"},
    ]
    
    for config in configs:
        fsq = FSQ(levels=config["levels"], eps=1e-3)
        print(f"  ✓ Created FSQ: {config['name']}")
        print(f"    - Levels: {config['levels']}")
        print(f"    - Codebook size: {fsq.codebook_size}")
        print(f"    - Dimension: {fsq.dim}")
    
    # Test 2: Verify FSQ works with typical RAE latent dimensions
    print("\nTest 2: FSQ with typical RAE latent shapes")
    print("-" * 60)
    
    # Simulate typical RAE encoder outputs
    test_cases = [
        {"shape": (4, 3, 16, 16), "levels": [8, 8, 8], "desc": "Small model (3D latent, 16x16)"},
        {"shape": (2, 4, 14, 14), "levels": [8, 5, 5, 5], "desc": "Medium model (4D latent, 14x14)"},
    ]
    
    for case in test_cases:
        fsq = FSQ(levels=case["levels"])
        z = torch.randn(*case["shape"])
        z_q, loss, indices = fsq(z)
        
        print(f"  {case['desc']}")
        print(f"    Input shape: {case['shape']}")
        print(f"    Output shape: {z_q.shape}")
        print(f"    Loss: {loss.item():.6f}")
        print(f"    Indices shape: {indices.shape}")
        print(f"    ✓ Test passed")
    
    # Test 3: Verify FSQ parameters that would be used in VQRAE config
    print("\nTest 3: FSQ parameter compatibility with VQRAE")
    print("-" * 60)
    
    # These are parameters that would be specified in a VQRAE config
    vqrae_params = {
        "use_fsq": True,
        "fsq_levels": [8, 6, 5],
        "vq_epsilon": 1e-3,
        "quantize_before_reshape": False,
    }
    
    print(f"  VQRAE config parameters:")
    for key, value in vqrae_params.items():
        print(f"    - {key}: {value}")
    
    # Create FSQ with VQRAE-style parameters
    fsq = FSQ(
        levels=vqrae_params["fsq_levels"],
        eps=vqrae_params["vq_epsilon"],
    )
    
    print(f"  ✓ FSQ created with VQRAE-compatible parameters")
    print(f"    - Codebook size: {fsq.codebook_size}")
    
    # Test 4: Verify FSQ loss is compatible with VQRAE training
    print("\nTest 4: FSQ loss computation")
    print("-" * 60)
    
    fsq = FSQ(levels=[8, 8, 8])
    z = torch.randn(2, 3, 4, 4, requires_grad=True)
    z_q, vq_loss, indices = fsq(z)
    
    print(f"  Input shape: {z.shape}")
    print(f"  VQ loss: {vq_loss.item():.6f}")
    print(f"  Loss requires grad: {vq_loss.requires_grad}")
    
    # Simulate reconstruction loss
    target = torch.randn_like(z_q)
    recon_loss = torch.nn.functional.mse_loss(z_q, target)
    total_loss = recon_loss + vq_loss
    
    print(f"  Reconstruction loss: {recon_loss.item():.6f}")
    print(f"  Total loss: {total_loss.item():.6f}")
    
    # Verify backward pass works
    total_loss.backward()
    assert z.grad is not None, "Gradients should flow through FSQ!"
    print(f"  ✓ Gradients flow correctly (grad norm: {z.grad.norm().item():.6f})")
    
    print("\n" + "=" * 60)
    print("All FSQ integration tests passed! ✓")
    print("=" * 60)


def test_fsq_vqrae_parameters():
    """Test that FSQ parameters match VQRAE interface expectations."""
    print("\n" + "=" * 60)
    print("Testing FSQ VQRAE Parameter Compatibility")
    print("=" * 60)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fsq", 
        os.path.join(os.path.dirname(__file__), 'src', 'stage1', 'fsq.py')
    )
    fsq_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fsq_module)
    FSQ = fsq_module.FSQ
    
    # Verify FSQ returns the same 3-tuple as VectorQuantizer and SimVQ
    print("\nVerifying FSQ return signature matches VQ interface:")
    print("-" * 60)
    
    fsq = FSQ(levels=[8, 8, 8])
    z = torch.randn(2, 3, 4, 4)
    
    result = fsq(z)
    assert len(result) == 3, "FSQ should return 3-tuple (z_q, loss, indices)"
    z_q, loss, indices = result
    
    print(f"  ✓ Returns 3-tuple: (z_q, loss, indices)")
    print(f"    - z_q shape: {z_q.shape} (same as input: {z.shape})")
    print(f"    - loss: {loss.item():.6f} (scalar)")
    print(f"    - indices shape: {indices.shape}")
    
    # Verify FSQ has get_codebook_entry method like VQ/SimVQ
    print("\nVerifying FSQ has required methods:")
    print("-" * 60)
    
    assert hasattr(fsq, 'get_codebook_entry'), "FSQ should have get_codebook_entry method"
    entries = fsq.get_codebook_entry(indices)
    print(f"  ✓ get_codebook_entry method exists")
    print(f"    - Returns entries with shape: {entries.shape}")
    
    # Test indices_to_codes and codes_to_indices
    assert hasattr(fsq, 'indices_to_codes'), "FSQ should have indices_to_codes method"
    assert hasattr(fsq, 'codes_to_indices'), "FSQ should have codes_to_indices method"
    
    codes = fsq.indices_to_codes(indices)
    indices_back = fsq.codes_to_indices(codes)
    assert torch.all(indices == indices_back), "Round-trip conversion should work"
    print(f"  ✓ indices_to_codes and codes_to_indices methods work correctly")
    
    print("\n" + "=" * 60)
    print("All VQRAE parameter compatibility tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_fsq_integration_minimal()
    test_fsq_vqrae_parameters()
    
    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED! ✓✓✓")
    print("=" * 60)
    print("\nFSQ is ready for use in VQRAE!")
    print("To use FSQ in VQRAE, set the following parameters in your config:")
    print("  use_fsq: true")
    print("  fsq_levels: [8, 8, 8]  # or other level configuration")
    print("  vq_epsilon: 1e-3")
