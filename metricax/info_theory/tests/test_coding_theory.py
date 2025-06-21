"""
Tests for Information Theory Coding Theory Functions

Comprehensive test suite covering:
- Optimal code length
- Fano inequality
- Redundancy

Each test includes:
- Basic functionality verification
- Mathematical properties validation
- Edge case handling
- Error condition testing
"""

import pytest
import math
from typing import List
import metricax.info_theory as it
from . import TEST_PRECISION


class TestOptimalCodeLength:
    """Test optimal code length function."""
    
    def test_uniform_distribution(self):
        """Test optimal code length for uniform distributions."""
        # Binary uniform
        p = [0.5, 0.5]
        lengths = it.optimal_code_length(p)
        expected = [1.0, 1.0]  # Each symbol needs 1 bit
        
        for actual, exp in zip(lengths, expected):
            assert abs(actual - exp) < TEST_PRECISION
    
    def test_known_values(self):
        """Test optimal code length for known analytical values."""
        # Skewed distribution
        p = [0.25, 0.75]
        lengths = it.optimal_code_length(p)
        
        # L*(x) = -log2(p(x))
        expected = [-math.log2(0.25), -math.log2(0.75)]
        
        for actual, exp in zip(lengths, expected):
            assert abs(actual - exp) < TEST_PRECISION
    
    def test_entropy_relationship(self):
        """Test that expected code length equals entropy."""
        p = [0.5, 0.3, 0.2]
        lengths = it.optimal_code_length(p)
        
        # Expected code length
        expected_length = sum(prob * length for prob, length in zip(p, lengths))
        
        # Should equal entropy
        entropy = it.entropy(p)
        assert abs(expected_length - entropy) < TEST_PRECISION
    
    def test_different_bases(self):
        """Test optimal code length with different bases."""
        p = [0.5, 0.5]
        
        # Binary codes (base 2)
        lengths_2 = it.optimal_code_length(p, base=2.0)
        expected_2 = [1.0, 1.0]
        
        for actual, exp in zip(lengths_2, expected_2):
            assert abs(actual - exp) < TEST_PRECISION
        
        # Ternary codes (base 3)
        lengths_3 = it.optimal_code_length(p, base=3.0)
        expected_3 = [-math.log(0.5) / math.log(3)] * 2
        
        for actual, exp in zip(lengths_3, expected_3):
            assert abs(actual - exp) < TEST_PRECISION
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single symbol (deterministic)
        p = [1.0]
        lengths = it.optimal_code_length(p)
        assert lengths[0] == 0.0  # No bits needed for certain outcome
        
        # Very small probabilities
        p = [0.999, 0.001]
        lengths = it.optimal_code_length(p)
        assert all(math.isfinite(length) for length in lengths)
    
    def test_input_validation(self):
        """Test error handling for invalid inputs."""
        # Invalid distribution
        with pytest.raises(ValueError):
            it.optimal_code_length([0.3, 0.3])  # Doesn't sum to 1
        
        # Invalid base
        with pytest.raises(ValueError):
            it.optimal_code_length([0.5, 0.5], base=0)
        
        with pytest.raises(ValueError):
            it.optimal_code_length([0.5, 0.5], base=1)


class TestFanoInequality:
    """Test Fano inequality function."""
    
    def test_perfect_channel(self):
        """Test Fano bound for perfect channel (no uncertainty)."""
        conditional_entropy = 0.0
        alphabet_size = 4
        
        bound = it.fano_inequality(conditional_entropy, alphabet_size)
        assert abs(bound - 0.0) < TEST_PRECISION
    
    def test_maximum_uncertainty(self):
        """Test Fano bound for maximum uncertainty."""
        alphabet_size = 4
        conditional_entropy = math.log2(alphabet_size)  # Maximum possible
        
        bound = it.fano_inequality(conditional_entropy, alphabet_size)
        
        # Should be close to 1 (maximum error probability)
        assert bound <= 1.0 + TEST_PRECISION
        assert bound >= 0.0
    
    def test_known_values(self):
        """Test Fano inequality for known analytical values."""
        conditional_entropy = 1.5
        alphabet_size = 8
        
        # Fano bound: Pe >= (H(X|Y) - 1) / log2(|X| - 1)
        expected = max(0.0, (conditional_entropy - 1) / math.log2(alphabet_size - 1))
        
        bound = it.fano_inequality(conditional_entropy, alphabet_size)
        assert abs(bound - expected) < TEST_PRECISION
    
    def test_monotonicity(self):
        """Test that Fano bound increases with conditional entropy."""
        alphabet_size = 4
        
        h1 = 0.5
        h2 = 1.0
        h3 = 1.5
        
        bound1 = it.fano_inequality(h1, alphabet_size)
        bound2 = it.fano_inequality(h2, alphabet_size)
        bound3 = it.fano_inequality(h3, alphabet_size)
        
        # Should be monotonically increasing
        assert bound1 <= bound2 + TEST_PRECISION
        assert bound2 <= bound3 + TEST_PRECISION
    
    def test_binary_case(self):
        """Test Fano inequality for binary alphabet."""
        conditional_entropy = 0.5
        alphabet_size = 2
        
        bound = it.fano_inequality(conditional_entropy, alphabet_size)
        
        # For binary case, should be well-defined
        assert 0.0 <= bound <= 1.0 + TEST_PRECISION
    
    def test_input_validation(self):
        """Test error handling for invalid inputs."""
        # Negative conditional entropy
        with pytest.raises(ValueError):
            it.fano_inequality(-0.1, 4)
        
        # Invalid alphabet size
        with pytest.raises(ValueError):
            it.fano_inequality(1.0, 1)  # Must be >= 2
        
        with pytest.raises(ValueError):
            it.fano_inequality(1.0, 0)


class TestRedundancy:
    """Test redundancy function."""
    
    def test_optimal_code(self):
        """Test redundancy for optimal code (should be 0)."""
        p = [0.5, 0.5]
        optimal_lengths = it.optimal_code_length(p)
        
        redundancy = it.redundancy(p, optimal_lengths)
        assert abs(redundancy - 0.0) < TEST_PRECISION
    
    def test_fixed_length_code(self):
        """Test redundancy for fixed-length code."""
        p = [0.8, 0.2]
        fixed_lengths = [1, 1]  # Fixed-length code
        
        redundancy = it.redundancy(p, fixed_lengths)
        
        # Expected length = 1.0, entropy ≈ 0.722
        expected_redundancy = 1.0 - it.entropy(p)
        assert abs(redundancy - expected_redundancy) < TEST_PRECISION
    
    def test_suboptimal_code(self):
        """Test redundancy for suboptimal code."""
        p = [0.5, 0.25, 0.25]
        suboptimal_lengths = [1, 2, 3]  # Suboptimal assignment
        
        redundancy = it.redundancy(p, suboptimal_lengths)
        
        # Should be positive (suboptimal)
        assert redundancy > -TEST_PRECISION
        
        # Expected length
        expected_length = sum(prob * length for prob, length in zip(p, suboptimal_lengths))
        entropy = it.entropy(p)
        expected_redundancy = expected_length - entropy
        
        assert abs(redundancy - expected_redundancy) < TEST_PRECISION
    
    def test_different_bases(self):
        """Test redundancy calculation with different bases."""
        p = [0.5, 0.3, 0.2]
        code_lengths = [1, 2, 2]
        
        # Base 2
        redundancy_2 = it.redundancy(p, code_lengths, base=2.0)
        
        # Base e (natural logarithm)
        redundancy_e = it.redundancy(p, code_lengths, base=math.e)
        
        # Both should be non-negative
        assert redundancy_2 >= -TEST_PRECISION
        assert redundancy_e >= -TEST_PRECISION
    
    def test_non_negativity(self):
        """Test that redundancy is always non-negative."""
        test_cases = [
            ([0.5, 0.5], [1, 1]),
            ([0.25, 0.75], [2, 1]),
            ([0.33, 0.33, 0.34], [2, 2, 2]),
            ([0.5, 0.25, 0.125, 0.125], [1, 2, 3, 3]),
        ]
        
        for p, lengths in test_cases:
            redundancy = it.redundancy(p, lengths)
            assert redundancy >= -TEST_PRECISION
    
    def test_kraft_inequality_violation(self):
        """Test redundancy when Kraft inequality is violated."""
        p = [0.5, 0.5]
        invalid_lengths = [0.5, 0.5]  # Fractional lengths (unusual but valid)
        
        # Should still compute redundancy
        redundancy = it.redundancy(p, invalid_lengths)
        assert math.isfinite(redundancy)
    
    def test_input_validation(self):
        """Test error handling for invalid inputs."""
        # Mismatched lengths
        with pytest.raises(ValueError):
            it.redundancy([0.5, 0.5], [1])
        
        # Negative code lengths
        with pytest.raises(ValueError):
            it.redundancy([0.5, 0.5], [-1, 1])
        
        # Invalid distribution
        with pytest.raises(ValueError):
            it.redundancy([0.3, 0.3], [1, 1])  # Doesn't sum to 1
        
        # Invalid base
        with pytest.raises(ValueError):
            it.redundancy([0.5, 0.5], [1, 1], base=0)


class TestCodingTheoryIntegration:
    """Integration tests for coding theory functions."""
    
    def test_shannon_coding_theorem(self):
        """Test Shannon's source coding theorem."""
        p = [0.5, 0.25, 0.125, 0.125]
        
        # Optimal code lengths
        optimal_lengths = it.optimal_code_length(p)
        
        # Expected code length should equal entropy
        expected_length = sum(prob * length for prob, length in zip(p, optimal_lengths))
        entropy = it.entropy(p)
        
        assert abs(expected_length - entropy) < TEST_PRECISION
        
        # Redundancy should be zero
        redundancy = it.redundancy(p, optimal_lengths)
        assert abs(redundancy - 0.0) < TEST_PRECISION
    
    def test_compression_efficiency(self):
        """Test compression efficiency analysis."""
        # Natural language-like distribution
        p = [0.12, 0.09, 0.08, 0.075, 0.07, 0.065, 0.06, 0.055,
             0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015,
             0.01, 0.005, 0.005, 0.005]
        
        # Fixed-length code
        fixed_length = math.ceil(math.log2(len(p)))
        fixed_lengths = [fixed_length] * len(p)
        
        # Optimal code
        optimal_lengths = it.optimal_code_length(p)
        
        # Compare redundancies
        fixed_redundancy = it.redundancy(p, fixed_lengths)
        optimal_redundancy = it.redundancy(p, optimal_lengths)
        
        # Optimal should have less redundancy
        assert optimal_redundancy <= fixed_redundancy + TEST_PRECISION
        assert optimal_redundancy <= TEST_PRECISION  # Should be ~0
    
    def test_channel_capacity_relationship(self):
        """Test relationship between Fano inequality and channel capacity."""
        # Simulate a binary symmetric channel
        error_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        for error_rate in error_rates:
            # Conditional entropy for BSC
            if error_rate == 0 or error_rate == 1:
                h_y_given_x = 0.0
            else:
                h_y_given_x = it.entropy([error_rate, 1 - error_rate])
            
            # Fano bound
            fano_bound = it.fano_inequality(h_y_given_x, 2)
            
            # Should be consistent with error rate
            if error_rate <= 0.5:
                assert fano_bound <= error_rate + 0.1  # Allow some tolerance


if __name__ == "__main__":
    pytest.main([__file__])
