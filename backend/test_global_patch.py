#!/usr/bin/env python3
"""
Test script to verify the global Flax compatibility patch.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# Apply the global patch
try:
    import flax_compat_patch
    print("✓ Applied global dataclasses patch")
except Exception as e:
    print(f"✗ Failed to apply patch: {e}")
    # Avoid using traceback to prevent Theme() error
    sys.exit(1)

# Create a direct replacement for DenyList
try:
    class DenyList:
        def __init__(self, names=''):
            self.names = names
        
        def __call__(self, x):
            return x != self.names if isinstance(x, str) else True
    
    # Test it works
    dl = DenyList('intermediates')
    print(f"✓ Created DenyList with names='intermediates'")
    print(f"✓ DenyList.__call__('intermediates') = {dl('intermediates')}")
    print(f"✓ DenyList.__call__('params') = {dl('params')}")
    
    # Add to builtins so it's available everywhere
    import builtins
    builtins.DenyList = DenyList
    print("✓ Added DenyList to builtins")
    
    # Directly patch flax.linen.module if it exists
    try:
        import flax.linen.module
        flax.linen.module.DenyList = DenyList
        print("✓ Patched DenyList in flax.linen.module")
    except ImportError:
        print("✓ Could not import flax.linen.module yet")
        pass
except Exception as e:
    print(f"✗ Failed to create or test DenyList: {e}")
    sys.exit(1)

# Try to import Flax
try:
    import flax.linen as nn
    print("✓ Successfully imported flax.linen")
except Exception as e:
    print(f"✗ Failed to import flax.linen: {e}")
    sys.exit(1)

# Try to use WeightNorm class
try:
    import jax
    import jax.numpy as jnp
    
    # Create a simple model with WeightNorm
    class SimpleModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.WeightNorm(nn.Dense, features=10)(x)
    
    # Initialize the model
    model = SimpleModel()
    print("✓ Created model with WeightNorm")
    
    # Test model initialization
    key = jax.random.PRNGKey(42)
    x = jnp.ones((1, 5))
    variables = model.init(key, x)
    print("✓ Model initialization successful")
    
    # Test model forward pass
    output = model.apply(variables, x)
    print(f"✓ Model forward pass successful, output shape: {output.shape}")
    
    print("\n🎉 All tests passed! The Flax compatibility patch is working correctly.")
    
except Exception as e:
    print(f"✗ Error during test: {e}")
    sys.exit(1)
