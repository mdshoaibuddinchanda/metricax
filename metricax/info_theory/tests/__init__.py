"""
Information Theory Tests

Comprehensive test suite for MetricaX Information Theory module.

Test modules:
- test_entropy: Tests for entropy measures and variants
- test_mutual_info: Tests for mutual information and dependence measures
- test_distance_measures: Tests for distribution distance metrics
- test_coding_theory: Tests for coding theory functions
- test_utils: Tests for utility functions
"""

# Test configuration
TEST_PRECISION = 1e-6  # Numerical precision for floating-point comparisons
TEST_TIMEOUT = 30      # Maximum test execution time in seconds

__all__ = [
    "TEST_PRECISION",
    "TEST_TIMEOUT",
]
