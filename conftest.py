"""
Repository pytest configuration helpers.

This conftest.py skips heavy/integration tests by default when required
dependencies (JAX/Flax or Hyperledger Fabric SDK) are not available.

Enable runs with environment variables:
 - RUN_INTEGRATION=1 to run integration tests
 - RUN_BLOCKCHAIN=1 to run blockchain tests (or install Fabric SDK)
 - RUN_JAX=1 to run JAX/Flax tests (or install jax/jaxlib/flax)

This makes it easier to run unit tests on developer machines or CI
without a full Fabric/JAX setup.
"""

import os
import pytest


def _has_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


RUN_INTEGRATION = os.getenv("RUN_INTEGRATION", "0") == "1"
RUN_BLOCKCHAIN = os.getenv("RUN_BLOCKCHAIN", "0") == "1"
RUN_JAX = os.getenv("RUN_JAX", "0") == "1"

HAS_JAX = _has_import("jax") and _has_import("flax")
HAS_FABRIC = _has_import("hfc") or _has_import("hyperledger")


def pytest_configure(config):
    # register markers to avoid warnings
    config.addinivalue_line("markers", "integration: mark test as integration (skipped by default)")
    config.addinivalue_line("markers", "blockchain: mark test that requires Hyperledger Fabric")
    config.addinivalue_line("markers", "jax: mark test that requires JAX/Flax")


def pytest_collection_modifyitems(config, items):
    """Modify collected tests to skip heavy ones unless enabled."""
    for item in items:
        p = str(item.fspath).replace('\\', '/')

        # Skip integration directory tests unless explicitly enabled
        if '/integration/' in p or 'integration' in item.nodeid.lower():
            if not RUN_INTEGRATION:
                item.add_marker(pytest.mark.skip(reason="Integration tests skipped (set RUN_INTEGRATION=1 to enable)"))
                continue

        # Skip blockchain-related tests if Fabric SDK not available and not enabled
        if '/blockchain/' in p or 'blockchain' in item.nodeid.lower():
            if not RUN_BLOCKCHAIN and not HAS_FABRIC:
                item.add_marker(pytest.mark.skip(reason="Blockchain tests skipped (set RUN_BLOCKCHAIN=1 or install Fabric SDK)"))
                continue

        # Skip JAX/Flax heavy tests if JAX not installed and not explicitly enabled
        if 'decision-service' in p or 'jax' in item.nodeid.lower() or 'flax' in item.nodeid.lower():
            if not RUN_JAX and not HAS_JAX:
                item.add_marker(pytest.mark.skip(reason="JAX/Flax tests skipped (set RUN_JAX=1 or install jax/jaxlib/flax)"))
                continue
