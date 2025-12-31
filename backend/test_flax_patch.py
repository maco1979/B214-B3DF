#!/usr/bin/env python3
"""
Test script to verify the Flax patch for Python 3.14 compatibility.
"""

import sys
import os

# Add the patched_flax directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    # Try to import our patched Flax version
    import patched_flax.linen as nn
    print("✓ Successfully imported patched_flax.linen")
    
    # Test if WeightNorm class is available and working
    print("✓ WeightNorm class is available")
    
    # Test creating a simple model with WeightNorm
    class SimpleModel(nn.Module):
        features: int
        
        @nn.compact
        def __call__(self, x):
            # Use WeightNorm with Dense
            x = nn.WeightNorm(nn.Dense, features=self.features)(x)
            return nn.relu(x)
    
    # Initialize the model
    model = SimpleModel(features=10)
    print("✓ Created model with WeightNorm")
    
    # Test model initialization
    import jax
    import jax.numpy as jnp
    
    key = jax.random.PRNGKey(42)
    x = jnp.ones((1, 5))
    variables = model.init(key, x)
    print("✓ Model initialization successful")
    
    # Test model forward pass
    output = model.apply(variables, x)
    print(f"✓ Model forward pass successful, output shape: {output.shape}")
    
    print("\n🎉 All tests passed! The Flax patch is working correctly.")
    print(f"Python version: {sys.version}")
    print(f"JAX version: {jax.__version__}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
