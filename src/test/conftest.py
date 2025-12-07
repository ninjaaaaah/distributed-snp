"""
Shared pytest fixtures and configuration for distributed-snp tests.

This file is automatically discovered by pytest and makes fixtures
available to all test files without explicit imports.
"""

import pytest
import sys
import os

# Add the main module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))


# Example: shared fixture for future use
# @pytest.fixture
# def common_config():
#     """Shared configuration for all tests."""
#     return {
#         'max_iterations': 1000,
#         'timeout': 10
#     }
