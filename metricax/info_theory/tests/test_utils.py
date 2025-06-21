"""
Tests for Information Theory Utility Functions

Comprehensive test suite covering:
- validate_distribution
- normalize_distribution
- joint_distribution
- safe_log

Each test includes:
- Basic functionality verification
- Edge case handling
- Error condition testing
- Numerical precision validation
"""

import pytest
import math
from typing import List
import metricax.info_theory as it
from . import TEST_PRECISION


class TestValidateDistribution:
    """Test validate_distribution function."""
    
    def test_valid_distributions(self):
        """Test validation of valid probability distributions."""
        valid_cases = [
            [0.5, 0.5],
            [0.25, 0.25, 0.25, 0.25],
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.1, 0.2, 0.3, 0.4],
        ]
        
        for dist in valid_cases:
            # Should not raise any exception
            it.validate_distribution(dist)
    
    def test_invalid_sum(self):
        """Test validation fails for distributions that don't sum to 1."""
        invalid_cases = [
            [0.3, 0.3],      # Sum = 0.6
            [0.5, 0.6],      # Sum = 1.1
            [0.25, 0.25, 0.25],  # Sum = 0.75
            [0.4, 0.4, 0.4], # Sum = 1.2
        ]
        
        for dist in invalid_cases:
            with pytest.raises(ValueError):
                it.validate_distribution(dist)
    
    def test_negative_probabilities(self):
        """Test validation fails for negative probabilities."""
        invalid_cases = [
            [-0.1, 1.1],
            [0.5, -0.5, 1.0],
            [-0.2, 0.6, 0.6],
        ]
        
        for dist in invalid_cases:
            with pytest.raises(ValueError):
                it.validate_distribution(dist)
    
    def test_non_finite_values(self):
        """Test validation fails for non-finite values."""
        invalid_cases = [
            [float('inf'), 0.0],
            [0.5, float('nan')],
            [float('-inf'), 1.0],
            [0.3, float('inf'), -float('inf')],
        ]
        
        for dist in invalid_cases:
            with pytest.raises(ValueError):
                it.validate_distribution(dist)
    
    def test_empty_distribution(self):
        """Test validation fails for empty distribution."""
        with pytest.raises(ValueError):
            it.validate_distribution([])
    
    def test_tolerance_parameter(self):
        """Test validation with different tolerance levels."""
        # Distribution that's slightly off
        almost_valid = [0.5, 0.5000001]  # Sum = 1.0000001
        
        # Should fail with strict tolerance
        with pytest.raises(ValueError):
            it.validate_distribution(almost_valid, tolerance=1e-9)
        
        # Should pass with loose tolerance
        it.validate_distribution(almost_valid, tolerance=1e-5)
    
    def test_single_element(self):
        """Test validation of single-element distributions."""
        # Valid single element
        it.validate_distribution([1.0])
        
        # Invalid single element
        with pytest.raises(ValueError):
            it.validate_distribution([0.5])


class TestNormalizeDistribution:
    """Test normalize_distribution function."""
    
    def test_already_normalized(self):
        """Test normalization of already valid distributions."""
        valid_dist = [0.3, 0.4, 0.3]
        normalized = it.normalize_distribution(valid_dist)
        
        # Should be unchanged
        for orig, norm in zip(valid_dist, normalized):
            assert abs(orig - norm) < TEST_PRECISION
        
        # Should sum to 1
        assert abs(sum(normalized) - 1.0) < TEST_PRECISION
    
    def test_unnormalized_positive(self):
        """Test normalization of positive unnormalized values."""
        unnormalized = [1, 2, 3]  # Sum = 6
        normalized = it.normalize_distribution(unnormalized)
        
        expected = [1/6, 2/6, 3/6]
        for exp, norm in zip(expected, normalized):
            assert abs(exp - norm) < TEST_PRECISION
        
        assert abs(sum(normalized) - 1.0) < TEST_PRECISION
    
    def test_different_scales(self):
        """Test normalization with different scales."""
        test_cases = [
            [10, 20, 30],           # Large integers
            [0.1, 0.2, 0.3],        # Small decimals
            [100, 200, 700],        # Mixed scale
            [1e-6, 2e-6, 3e-6],     # Very small numbers
        ]
        
        for unnormalized in test_cases:
            normalized = it.normalize_distribution(unnormalized)
            
            # Should sum to 1
            assert abs(sum(normalized) - 1.0) < TEST_PRECISION
            
            # Should preserve ratios
            total = sum(unnormalized)
            for orig, norm in zip(unnormalized, normalized):
                expected = orig / total
                assert abs(norm - expected) < TEST_PRECISION
    
    def test_zero_handling(self):
        """Test normalization with zero values."""
        with_zeros = [1, 0, 2, 0, 3]
        normalized = it.normalize_distribution(with_zeros)
        
        expected = [1/6, 0, 2/6, 0, 3/6]
        for exp, norm in zip(expected, normalized):
            assert abs(exp - norm) < TEST_PRECISION
    
    def test_all_zeros(self):
        """Test normalization fails for all-zero input."""
        with pytest.raises(ValueError):
            it.normalize_distribution([0, 0, 0])
    
    def test_negative_values(self):
        """Test normalization fails for negative values."""
        with pytest.raises(ValueError):
            it.normalize_distribution([1, -1, 2])
    
    def test_single_element(self):
        """Test normalization of single element."""
        normalized = it.normalize_distribution([5])
        assert abs(normalized[0] - 1.0) < TEST_PRECISION


class TestJointDistribution:
    """Test joint_distribution function."""
    
    def test_independent_variables(self):
        """Test joint distribution for independent variables."""
        p_x = [0.6, 0.4]
        p_y = [0.3, 0.7]
        
        joint = it.joint_distribution(p_x, p_y)
        
        # For independent variables: P(X,Y) = P(X) * P(Y)
        expected = [
            [0.6 * 0.3, 0.6 * 0.7],  # X=0
            [0.4 * 0.3, 0.4 * 0.7],  # X=1
        ]
        
        for i in range(len(expected)):
            for j in range(len(expected[i])):
                assert abs(joint[i][j] - expected[i][j]) < TEST_PRECISION
    
    def test_conditional_distribution(self):
        """Test joint distribution from conditional probabilities."""
        p_x = [0.5, 0.5]
        p_y_given_x = [
            [0.8, 0.2],  # P(Y|X=0)
            [0.3, 0.7],  # P(Y|X=1)
        ]
        
        joint = it.joint_distribution(p_x, p_y_given_x)
        
        # P(X,Y) = P(Y|X) * P(X)
        expected = [
            [0.5 * 0.8, 0.5 * 0.2],  # X=0
            [0.5 * 0.3, 0.5 * 0.7],  # X=1
        ]
        
        for i in range(len(expected)):
            for j in range(len(expected[i])):
                assert abs(joint[i][j] - expected[i][j]) < TEST_PRECISION
    
    def test_marginal_consistency(self):
        """Test that marginals are consistent with joint distribution."""
        p_x = [0.3, 0.7]
        p_y_given_x = [
            [0.6, 0.4],  # P(Y|X=0)
            [0.2, 0.8],  # P(Y|X=1)
        ]
        
        joint = it.joint_distribution(p_x, p_y_given_x)
        
        # Check X marginal
        x_marginal = [sum(joint[i]) for i in range(len(joint))]
        for orig, computed in zip(p_x, x_marginal):
            assert abs(orig - computed) < TEST_PRECISION
        
        # Check Y marginal
        y_marginal = [sum(joint[i][j] for i in range(len(joint))) 
                     for j in range(len(joint[0]))]
        assert abs(sum(y_marginal) - 1.0) < TEST_PRECISION
    
    def test_probability_sum(self):
        """Test that joint distribution sums to 1."""
        p_x = [0.25, 0.75]
        p_y_given_x = [
            [0.4, 0.6],
            [0.8, 0.2],
        ]
        
        joint = it.joint_distribution(p_x, p_y_given_x)
        
        total = sum(sum(row) for row in joint)
        assert abs(total - 1.0) < TEST_PRECISION
    
    def test_input_validation(self):
        """Test error handling for invalid inputs."""
        # Invalid marginal distribution
        with pytest.raises(ValueError):
            it.joint_distribution([0.3, 0.3], [[0.5, 0.5], [0.5, 0.5]])
        
        # Mismatched dimensions
        with pytest.raises(ValueError):
            it.joint_distribution([0.5, 0.5], [[0.5, 0.5]])  # Only one conditional
        
        # Invalid conditional distributions
        with pytest.raises(ValueError):
            it.joint_distribution([0.5, 0.5], [[0.3, 0.3], [0.5, 0.5]])


class TestSafeLog:
    """Test safe_log function."""
    
    def test_positive_values(self):
        """Test safe logarithm for positive values."""
        test_cases = [
            (1.0, 2.0, 0.0),        # log2(1) = 0
            (2.0, 2.0, 1.0),        # log2(2) = 1
            (8.0, 2.0, 3.0),        # log2(8) = 3
            (math.e, math.e, 1.0),  # ln(e) = 1
            (10.0, 10.0, 1.0),      # log10(10) = 1
        ]
        
        for x, base, expected in test_cases:
            result = it.safe_log(x, base)
            assert abs(result - expected) < TEST_PRECISION
    
    def test_zero_and_negative(self):
        """Test safe logarithm for zero and negative values."""
        # Zero should use epsilon
        result = it.safe_log(0.0, 2.0)
        assert math.isfinite(result)
        assert result < 0  # Should be very negative
        
        # Negative should use epsilon
        result = it.safe_log(-1.0, 2.0)
        assert math.isfinite(result)
        assert result < 0  # Should be very negative
    
    def test_custom_epsilon(self):
        """Test safe logarithm with custom epsilon values."""
        epsilon_values = [1e-10, 1e-15, 1e-20]
        
        for epsilon in epsilon_values:
            result = it.safe_log(0.0, 2.0, epsilon)
            expected = math.log(epsilon) / math.log(2.0)
            assert abs(result - expected) < TEST_PRECISION
    
    def test_different_bases(self):
        """Test safe logarithm with different bases."""
        x = 0.5
        bases = [2.0, math.e, 10.0, 3.0]
        
        for base in bases:
            result = it.safe_log(x, base)
            expected = math.log(x) / math.log(base)
            assert abs(result - expected) < TEST_PRECISION
    
    def test_very_small_values(self):
        """Test safe logarithm for very small positive values."""
        small_values = [1e-10, 1e-15, 1e-20]
        
        for x in small_values:
            result = it.safe_log(x, 2.0)
            assert math.isfinite(result)
            assert result < 0  # Should be negative
    
    def test_large_values(self):
        """Test safe logarithm for large values."""
        large_values = [1e10, 1e15, 1e20]
        
        for x in large_values:
            result = it.safe_log(x, 2.0)
            assert math.isfinite(result)
            assert result > 0  # Should be positive
    
    def test_base_validation(self):
        """Test error handling for invalid bases."""
        # Base must be positive and not equal to 1
        invalid_bases = [0.0, -1.0, 1.0]
        
        for base in invalid_bases:
            with pytest.raises(ValueError):
                it.safe_log(2.0, base)


class TestUtilsIntegration:
    """Integration tests for utility functions."""
    
    def test_entropy_calculation_pipeline(self):
        """Test complete pipeline using utility functions."""
        # Start with unnormalized data
        raw_counts = [10, 20, 30, 40]
        
        # Normalize to probability distribution
        p = it.normalize_distribution(raw_counts)
        
        # Validate the result
        it.validate_distribution(p)
        
        # Calculate entropy using safe_log
        entropy_manual = -sum(prob * it.safe_log(prob, 2.0) for prob in p if prob > 0)
        entropy_builtin = it.entropy(p)
        
        assert abs(entropy_manual - entropy_builtin) < TEST_PRECISION
    
    def test_joint_distribution_analysis(self):
        """Test joint distribution analysis workflow."""
        # Create marginal and conditional distributions
        p_x = it.normalize_distribution([3, 7])  # Normalize raw counts
        p_y_given_x = [
            it.normalize_distribution([4, 6]),   # P(Y|X=0)
            it.normalize_distribution([8, 2]),   # P(Y|X=1)
        ]
        
        # Create joint distribution
        joint = it.joint_distribution(p_x, p_y_given_x)
        
        # Flatten for entropy calculation
        joint_flat = [joint[i][j] for i in range(len(joint)) for j in range(len(joint[0]))]
        
        # Validate and calculate entropy
        it.validate_distribution(joint_flat)
        joint_entropy = it.entropy(joint_flat)
        
        assert joint_entropy > 0
        assert math.isfinite(joint_entropy)
    
    def test_error_handling_consistency(self):
        """Test that error handling is consistent across utility functions."""
        invalid_dist = [0.3, 0.3]  # Doesn't sum to 1
        
        # All functions should handle invalid distributions consistently
        with pytest.raises(ValueError):
            it.validate_distribution(invalid_dist)
        
        # Functions that use validate_distribution should also fail
        with pytest.raises(ValueError):
            it.entropy(invalid_dist)


if __name__ == "__main__":
    pytest.main([__file__])
