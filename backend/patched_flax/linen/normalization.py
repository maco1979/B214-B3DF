# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalization layers.

This module implements normalization layers like BatchNorm, LayerNorm, etc.
"""

# Import the original normalization module
import flax.linen.normalization as _original_normalization
from flax.linen.module import Module
from flax.linen.initializers import constant, zeros, ones
from flax.linen.dtypes import promote_dtype
from flax.linen.compat import get_module_parent

import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional, Tuple, Type, Union

# Re-export everything from the original module
for name in dir(_original_normalization):
  if not name.startswith('_'):
    globals()[name] = getattr(_original_normalization, name)

# Patch for WeightNorm class to fix Python 3.14 compatibility
class WeightNorm(Module):
  """Weight normalization wrapper for linear operations.

  This wrapper reparameterizes a linear operation by separating the magnitude
  and direction of the weight vector.
  See https://arxiv.org/abs/1602.07868 for more details.

  Attributes:
    module: Linear module to wrap (e.g. Dense, Conv).
    use_bias: Whether to add a bias term.
    param_dtype: Dtype of the parameters.
    kernel_init: Initializer for the kernel.
    bias_init: Initializer for the bias.
    variable_filter: A function that takes a variable name and returns True if
      the variable should be normalized.
  """
  module: Type[Module]
  use_bias: bool = True
  param_dtype: Optional[jnp.dtype] = None
  kernel_init: Callable = ones
  bias_init: Callable = zeros
  variable_filter: Optional[Callable[[str], bool]] = None  # Fixed: Added type annotation

  @property
  def wrapped_module(self):
    """Return the wrapped module class."""
    return self.module

  def setup(self):
    """Set up the weight norm parameters."""
    # Create the wrapped module
    self._module = self.module(
        use_bias=False,  # We'll add our own bias if needed
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
    )

    if self.use_bias:
      self.bias = self.param('bias', self.bias_init, (self._module.features,))

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Apply the weight normalized module."""
    # Get the weight variable from the wrapped module
    variables = self._module.variables
    kernel = variables['params']['kernel']

    # Apply weight normalization
    norm = jnp.linalg.norm(kernel, axis=0, keepdims=True)
    normalized_kernel = kernel / (norm + 1e-8)

    # Call the wrapped module with normalized weight
    variables['params']['kernel'] = normalized_kernel
    output = self._module.apply(variables, inputs)

    # Add bias if needed
    if self.use_bias:
      output = output + jnp.broadcast_to(self.bias, output.shape)

    return output

# Make sure our patched version is used
__all__ = [
    *[name for name in dir(_original_normalization) if not name.startswith('_')],
    'WeightNorm',
]
