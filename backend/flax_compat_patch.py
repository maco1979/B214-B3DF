#!/usr/bin/env python3
"""
Global compatibility patch for Flax 0.12.2 with Python 3.14.

This patch modifies the dataclasses module to address two issues:
1. Missing type annotations (required for Flax with Python 3.14)
2. Non-default arguments following default arguments (common in older Flax code)
3. Replace DenyList class to fix initialization issues
4. Fix Theme() initialization in Python 3.14's _colorize module
"""

import sys
import types
from typing import Any, Dict, List, Callable, Optional

# Only apply the patch for Python 3.14+
if sys.version_info >= (3, 14):
    # First, fix the Theme() issue in Python 3.14's _colorize module
    try:
        import _colorize
        if hasattr(_colorize, 'Theme'):
            # Patch the Theme class to accept any arguments
            original_theme = _colorize.Theme
            class PatchedTheme:
                def __init__(self, *args, **kwargs):
                    # Ignore all arguments
                    pass
                
                def no_colors(self):
                    return self
            
            _colorize.Theme = PatchedTheme
            _colorize.default_theme = PatchedTheme()
            _colorize.theme_no_color = PatchedTheme()
            print("✓ Fixed Theme() initialization in _colorize module")
    except Exception as e:
        print(f"✗ Failed to fix Theme() issue: {e}")
    
    # Add patched_flax directory to sys.path so it's found first
    import os
    patched_flax_path = os.path.join(os.path.dirname(__file__), 'patched_flax')
    if patched_flax_path not in sys.path:
        sys.path.insert(0, patched_flax_path)
    
    # Get the dataclasses module
    import dataclasses
    
    # Save the original functions
    original_process_class = dataclasses._process_class
    original_init_fn = dataclasses._init_fn
    
    def patched_process_class(cls, init, repr, eq, order, unsafe_hash, 
                            frozen, match_args, kw_only, slots, 
                            weakref_slot):
        """Patched version of dataclasses._process_class that ignores missing type annotations."""
        
        # Ensure __annotations__ exists
        if '__annotations__' not in cls.__dict__:
            cls.__annotations__ = {}
        
        # Get all attributes that might be fields
        potential_fields = {}
        for name in dir(cls):
            if name.startswith('_'):
                continue
            value = getattr(cls, name)
            if isinstance(value, dataclasses.Field):
                potential_fields[name] = value
        
        # Add type annotations for fields that are missing them
        for name, field in potential_fields.items():
            if name not in cls.__annotations__:
                # Add Any type annotation for missing fields
                cls.__annotations__[name] = Any
        
        # Call the original _process_class function
        return original_process_class(cls, init, repr, eq, order, unsafe_hash, 
                                     frozen, match_args, kw_only, slots, 
                                     weakref_slot)
    
    def patched_init_fn(fields, std_fields, kw_only_fields, frozen, 
                      has_post_init, self_name, func_builder, slots):
        """Patched version of dataclasses._init_fn that bypasses non-default arg order errors."""
        try:
            # Create a simple init function that handles argument order
            def simple_init(self, *args, **kwargs):
                # Create a mapping from field name to field
                field_dict = {name: field for name, field in fields}
                
                # Assign positional arguments
                for i, arg in enumerate(args):
                    if i < len(fields):
                        name, _ = fields[i]
                        setattr(self, name, arg)
                
                # Assign keyword arguments
                for name, value in kwargs.items():
                    if name in field_dict:
                        setattr(self, name, value)
                
                # Assign default values for remaining fields
                for name, field in fields:
                    if not hasattr(self, name):
                        if field.default is not dataclasses.MISSING:
                            setattr(self, name, field.default)
                        elif field.default_factory is not dataclasses.MISSING:
                            setattr(self, name, field.default_factory())
            
            return simple_init
        except Exception as e:
            # If all else fails, use the original init
            return original_init_fn(fields, std_fields, kw_only_fields, frozen, 
                                  has_post_init, self_name, func_builder, slots)
    
    # Replace the original functions with our patched versions
    dataclasses._process_class = patched_process_class
    dataclasses._init_fn = patched_init_fn
    
    print("✓ Applied dataclasses patch for Python 3.14 compatibility")
    print("✓ Applied init_fn patch for non-default argument order")
    print("✓ Added patched_flax to Python path")
else:
    print("✓ No patch needed (Python < 3.14)")