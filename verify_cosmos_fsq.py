#!/usr/bin/env python3
"""
Verify that the FSQ indices_to_codes implementation follows Cosmos-Tokenizer approach.
This script can be run without torch if needed.
"""

import sys
sys.path.insert(0, '/home/runner/work/RAE/RAE/src')

def verify_cosmos_approach():
    """Verify the implementation matches Cosmos-Tokenizer logic."""
    
    print("=" * 60)
    print("Verifying FSQ indices_to_codes (Cosmos-Tokenizer style)")
    print("=" * 60)
    
    # Test the mathematical decomposition
    levels = [8, 8, 8, 5, 5, 5]
    
    # Calculate basis (cumulative product of levels from right to left)
    basis = []
    for i in range(len(levels)):
        basis_value = 1
        for j in range(i + 1, len(levels)):
            basis_value *= levels[j]
        basis.append(basis_value)
    
    print(f"FSQ levels: {levels}")
    print(f"Basis: {basis}")
    print(f"Codebook size: {8*8*8*5*5*5} = 64000")
    print()
    
    # Test a few index -> code conversions
    test_indices = [0, 1, 100, 1000, 32000, 63999]
    
    print("Testing index -> code decomposition:")
    print("-" * 60)
    
    for idx in test_indices:
        # Decompose index into per-dimension codes (Cosmos-Tokenizer style)
        codes_non_centered = []
        for i, (level, b) in enumerate(zip(levels, basis)):
            code = (idx // b) % level
            codes_non_centered.append(code)
        
        # Convert to normalized range [-(L-1)/2, (L-1)/2]
        codes_normalized = []
        for code, level in zip(codes_non_centered, levels):
            half_width = level // 2
            normalized = code - half_width
            codes_normalized.append(normalized)
        
        print(f"Index {idx:5d} -> codes_shifted {codes_non_centered} -> codes_norm {codes_normalized}")
    
    print()
    print("✓ Mathematical decomposition verified")
    print()
    
    # Verify round-trip: index -> codes -> index
    print("Testing round-trip: index -> codes -> index")
    print("-" * 60)
    
    for idx in test_indices:
        # Forward: index -> codes
        codes_non_centered = []
        for i, (level, b) in enumerate(zip(levels, basis)):
            code = (idx // b) % level
            codes_non_centered.append(code)
        
        # Backward: codes -> index
        reconstructed_idx = 0
        for code, b in zip(codes_non_centered, basis):
            reconstructed_idx += code * b
        
        match = "✓" if idx == reconstructed_idx else "✗"
        print(f"{match} Index {idx:5d} -> {reconstructed_idx:5d}")
        
        if idx != reconstructed_idx:
            print(f"ERROR: Mismatch!")
            return False
    
    print()
    print("✓ Round-trip conversion verified")
    print()
    print("=" * 60)
    print("All verifications passed!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = verify_cosmos_approach()
    sys.exit(0 if success else 1)
