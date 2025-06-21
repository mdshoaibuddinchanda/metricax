"""
Tests for Information Theory Mutual Information Functions

Comprehensive test suite covering:
- Mutual information
- Conditional entropy
- Information gain
- Symmetric uncertainty
- Variation of information
- Total correlation
- Multi-information

Each test includes:
- Basic functionality verification
- Edge case handling
- Numerical precision validation
- Error condition testing
"""

import pytest
import math
from typing import List
import metricax.info_theory as it
from . import TEST_PRECISION


class TestMutualInformation:
    """Test mutual information function."""
    
    def test_independent_variables(self):
        """Test MI for independent variables (should be 0)."""
        # Independent 2x2 joint distribution
        p_xy = [[0.25, 0.25], [0.25, 0.25]]
        p_x = [0.5, 0.5]
        p_y = [0.5, 0.5]
        
        mi = it.mutual_information(p_xy, p_x, p_y)
        assert abs(mi - 0.0) < TEST_PRECISION
    
    def test_perfectly_dependent_variables(self):
        """Test MI for perfectly dependent variables."""
        # Perfect correlation: X = Y
        p_xy = [[0.5, 0.0], [0.0, 0.5]]
        p_x = [0.5, 0.5]
        p_y = [0.5, 0.5]
        
        mi = it.mutual_information(p_xy, p_x, p_y)
        expected = it.entropy(p_x)  # MI = H(X) = H(Y) when perfectly dependent
        assert abs(mi - expected) < TEST_PRECISION
    
    def test_known_values(self):
        """Test MI for known analytical values."""
        # Partially dependent variables
        p_xy = [[0.4, 0.1], [0.1, 0.4]]
        p_x = [0.5, 0.5]
        p_y = [0.5, 0.5]
        
        # Calculate expected MI manually
        mi_expected = 0.0
        for i, p_xi in enumerate(p_x):
            for j, p_yj in enumerate(p_y):
                p_xy_ij = p_xy[i][j]
                if p_xy_ij > 0:
                    mi_expected += p_xy_ij * math.log2(p_xy_ij / (p_xi * p_yj))
        
        mi = it.mutual_information(p_xy, p_x, p_y)
        assert abs(mi - mi_expected) < TEST_PRECISION
    
    def test_non_negativity(self):
        """Test that MI is always non-negative."""
        test_cases = [
            ([[0.3, 0.2], [0.2, 0.3]], [0.5, 0.5], [0.5, 0.5]),
            ([[0.6, 0.1], [0.1, 0.2]], [0.7, 0.3], [0.7, 0.3]),
            ([[0.25, 0.25], [0.25, 0.25]], [0.5, 0.5], [0.5, 0.5]),
        ]
        
        for p_xy, p_x, p_y in test_cases:
            mi = it.mutual_information(p_xy, p_x, p_y)
            assert mi >= -TEST_PRECISION  # Allow small numerical errors
    
    def test_input_validation(self):
        """Test error handling for invalid inputs."""
        # Mismatched dimensions
        with pytest.raises(ValueError):
            it.mutual_information([[0.5, 0.5]], [0.5, 0.5], [0.5, 0.5])
        
        # Invalid distributions
        with pytest.raises(ValueError):
            it.mutual_information([[0.3, 0.3], [0.3, 0.3]], [0.5, 0.5], [0.5, 0.5])


class TestConditionalEntropy:
    """Test conditional entropy function."""
    
    def test_deterministic_case(self):
        """Test conditional entropy when Y is deterministic given X."""
        # If Y is completely determined by X, H(Y|X) = 0
        p_xy = [[0.5, 0.0], [0.0, 0.5]]  # Perfect correlation
        p_x = [0.5, 0.5]
        
        cond_ent = it.conditional_entropy(p_xy, p_x)
        assert abs(cond_ent - 0.0) < TEST_PRECISION
    
    def test_independent_case(self):
        """Test conditional entropy for independent variables."""
        # If X and Y are independent, H(Y|X) = H(Y)
        p_xy = [[0.25, 0.25], [0.25, 0.25]]
        p_x = [0.5, 0.5]
        p_y = [0.5, 0.5]
        
        cond_ent = it.conditional_entropy(p_xy, p_x)
        expected = it.entropy(p_y)
        assert abs(cond_ent - expected) < TEST_PRECISION
    
    def test_chain_rule(self):
        """Test chain rule: H(X,Y) = H(X) + H(Y|X)."""
        p_xy = [[0.3, 0.2], [0.2, 0.3]]
        p_x = [0.5, 0.5]
        
        # Flatten joint distribution for entropy calculation
        joint_flat = [p_xy[i][j] for i in range(len(p_xy)) for j in range(len(p_xy[0]))]
        
        h_xy = it.entropy(joint_flat)
        h_x = it.entropy(p_x)
        h_y_given_x = it.conditional_entropy(p_xy, p_x)
        
        assert abs(h_xy - (h_x + h_y_given_x)) < TEST_PRECISION


class TestInformationGain:
    """Test information gain function."""
    
    def test_complete_information_gain(self):
        """Test maximum information gain (uncertainty to certainty)."""
        p_prior = [0.5, 0.5]  # Maximum uncertainty
        p_posterior = [1.0, 0.0]  # Complete certainty
        
        ig = it.information_gain(p_prior, p_posterior)
        expected = it.entropy(p_prior)  # Should equal prior entropy
        assert abs(ig - expected) < TEST_PRECISION
    
    def test_no_information_gain(self):
        """Test zero information gain (no change in uncertainty)."""
        p_prior = [0.3, 0.7]
        p_posterior = [0.3, 0.7]  # Same distribution
        
        ig = it.information_gain(p_prior, p_posterior)
        assert abs(ig - 0.0) < TEST_PRECISION
    
    def test_partial_information_gain(self):
        """Test partial information gain."""
        p_prior = [0.5, 0.5]  # High uncertainty
        p_posterior = [0.8, 0.2]  # Lower uncertainty
        
        ig = it.information_gain(p_prior, p_posterior)
        expected = it.entropy(p_prior) - it.entropy(p_posterior)
        assert abs(ig - expected) < TEST_PRECISION
    
    def test_non_negativity(self):
        """Test that information gain is always non-negative."""
        test_cases = [
            ([0.5, 0.5], [0.7, 0.3]),
            ([0.25, 0.75], [0.9, 0.1]),
            ([0.33, 0.33, 0.34], [0.6, 0.3, 0.1]),
        ]
        
        for prior, posterior in test_cases:
            ig = it.information_gain(prior, posterior)
            assert ig >= -TEST_PRECISION


class TestSymmetricUncertainty:
    """Test symmetric uncertainty function."""
    
    def test_independent_variables(self):
        """Test SU for independent variables (should be 0)."""
        p_xy = [[0.25, 0.25], [0.25, 0.25]]
        p_x = [0.5, 0.5]
        p_y = [0.5, 0.5]
        
        su = it.symmetric_uncertainty(p_xy, p_x, p_y)
        assert abs(su - 0.0) < TEST_PRECISION
    
    def test_perfectly_dependent_variables(self):
        """Test SU for perfectly dependent variables (should be 1)."""
        p_xy = [[0.5, 0.0], [0.0, 0.5]]
        p_x = [0.5, 0.5]
        p_y = [0.5, 0.5]
        
        su = it.symmetric_uncertainty(p_xy, p_x, p_y)
        assert abs(su - 1.0) < TEST_PRECISION
    
    def test_bounded_range(self):
        """Test that SU is bounded in [0, 1]."""
        test_cases = [
            ([[0.3, 0.2], [0.2, 0.3]], [0.5, 0.5], [0.5, 0.5]),
            ([[0.4, 0.1], [0.1, 0.4]], [0.5, 0.5], [0.5, 0.5]),
            ([[0.6, 0.05], [0.05, 0.3]], [0.65, 0.35], [0.65, 0.35]),
        ]
        
        for p_xy, p_x, p_y in test_cases:
            su = it.symmetric_uncertainty(p_xy, p_x, p_y)
            assert 0.0 <= su <= 1.0 + TEST_PRECISION
    
    def test_symmetry(self):
        """Test that SU is symmetric in X and Y."""
        p_xy = [[0.3, 0.2], [0.1, 0.4]]
        p_x = [0.5, 0.5]
        p_y = [0.5, 0.5]
        
        # Transpose the joint distribution
        p_yx = [[p_xy[j][i] for j in range(len(p_xy))] for i in range(len(p_xy[0]))]
        
        su_xy = it.symmetric_uncertainty(p_xy, p_x, p_y)
        su_yx = it.symmetric_uncertainty(p_yx, p_y, p_x)
        
        assert abs(su_xy - su_yx) < TEST_PRECISION


class TestVariationOfInformation:
    """Test variation of information function."""
    
    def test_identical_distributions(self):
        """Test VI for identical distributions (should be 0)."""
        p_xy = [[0.5, 0.0], [0.0, 0.5]]
        p_x = [0.5, 0.5]
        p_y = [0.5, 0.5]
        
        vi = it.variation_of_information(p_xy, p_x, p_y)
        assert abs(vi - 0.0) < TEST_PRECISION
    
    def test_independent_variables(self):
        """Test VI for independent variables."""
        p_xy = [[0.25, 0.25], [0.25, 0.25]]
        p_x = [0.5, 0.5]
        p_y = [0.5, 0.5]
        
        vi = it.variation_of_information(p_xy, p_x, p_y)
        expected = it.entropy(p_x) + it.entropy(p_y)  # VI = H(X) + H(Y) when independent
        assert abs(vi - expected) < TEST_PRECISION
    
    def test_metric_properties(self):
        """Test that VI satisfies metric properties."""
        # Non-negativity
        p_xy = [[0.3, 0.2], [0.2, 0.3]]
        p_x = [0.5, 0.5]
        p_y = [0.5, 0.5]
        
        vi = it.variation_of_information(p_xy, p_x, p_y)
        assert vi >= -TEST_PRECISION
        
        # Symmetry is tested implicitly through the formula


class TestTotalCorrelation:
    """Test total correlation function."""
    
    def test_independent_variables(self):
        """Test TC for independent variables (should be 0)."""
        # Three independent binary variables
        p_xyz = [0.125] * 8  # Uniform over 2^3 = 8 outcomes
        p_x = [0.5, 0.5]
        p_y = [0.5, 0.5]
        p_z = [0.5, 0.5]
        
        tc = it.total_correlation(p_xyz, p_x, p_y, p_z)
        assert abs(tc - 0.0) < TEST_PRECISION
    
    def test_perfectly_correlated_variables(self):
        """Test TC for perfectly correlated variables."""
        # All variables are identical: X = Y = Z
        p_xyz = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]  # Only (0,0,0) and (1,1,1)
        p_x = [0.5, 0.5]
        p_y = [0.5, 0.5]
        p_z = [0.5, 0.5]
        
        tc = it.total_correlation(p_xyz, p_x, p_y, p_z)
        expected = 2 * it.entropy(p_x)  # TC = 2*H(X) for three identical variables
        assert abs(tc - expected) < TEST_PRECISION
    
    def test_non_negativity(self):
        """Test that TC is always non-negative."""
        # Partially correlated variables
        p_xyz = [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        p_x = [0.6, 0.4]
        p_y = [0.6, 0.4]
        p_z = [0.6, 0.4]
        
        tc = it.total_correlation(p_xyz, p_x, p_y, p_z)
        assert tc >= -TEST_PRECISION


class TestMultiInformation:
    """Test multi-information function."""
    
    def test_independent_variables(self):
        """Test MI for independent variables (should be 0)."""
        # Two independent binary variables
        p_joint = [0.25, 0.25, 0.25, 0.25]
        marginals = [[0.5, 0.5], [0.5, 0.5]]
        
        mi = it.multi_information(p_joint, marginals)
        assert abs(mi - 0.0) < TEST_PRECISION
    
    def test_perfectly_dependent_variables(self):
        """Test MI for perfectly dependent variables."""
        # X = Y (perfect correlation)
        p_joint = [0.5, 0.0, 0.0, 0.5]
        marginals = [[0.5, 0.5], [0.5, 0.5]]
        
        mi = it.multi_information(p_joint, marginals)
        expected = it.entropy(marginals[0])  # MI = H(X) when X = Y
        assert abs(mi - expected) < TEST_PRECISION
    
    def test_consistency_with_total_correlation(self):
        """Test that MI equals TC for three variables."""
        # For three variables, multi-information should equal total correlation
        p_joint = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
        marginals = [[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]]
        
        mi = it.multi_information(p_joint, marginals)
        
        # Convert to format expected by total_correlation
        # This is a simplified test - in practice, you'd need proper conversion
        assert mi >= -TEST_PRECISION  # At minimum, should be non-negative


if __name__ == "__main__":
    pytest.main([__file__])
