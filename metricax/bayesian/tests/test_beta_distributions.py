"""
Unit tests for metricax.bayesian.beta_distributions module.
"""

import pytest
import math
import sys
import os

# Add the parent directory to the path to import metricax
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import metricax.bayesian as mb


class TestBetaPdf:
    """Test cases for beta_pdf"""
    
    def test_beta_pdf_uniform(self):
        """Test Beta(1,1) = Uniform distribution"""
        # Beta(1,1) should be uniform on [0,1] with PDF = 1
        assert abs(mb.beta_pdf(0.5, 1, 1) - 1.0) < 1e-10
        assert abs(mb.beta_pdf(0.1, 1, 1) - 1.0) < 1e-10
        assert abs(mb.beta_pdf(0.9, 1, 1) - 1.0) < 1e-10
    
    def test_beta_pdf_symmetric(self):
        """Test symmetric Beta distribution"""
        # Beta(2,2) should be symmetric around 0.5
        pdf_at_half = mb.beta_pdf(0.5, 2, 2)
        assert pdf_at_half > 0
        
        # Should be symmetric
        assert abs(mb.beta_pdf(0.3, 2, 2) - mb.beta_pdf(0.7, 2, 2)) < 1e-10
    
    def test_beta_pdf_boundary_cases(self):
        """Test boundary cases"""
        # At x=0 and x=1
        assert mb.beta_pdf(0.0, 2, 2) == 0.0
        assert mb.beta_pdf(1.0, 2, 2) == 0.0
        
        # Beta(1,1) at boundaries
        assert mb.beta_pdf(0.0, 1, 1) == 1.0
        assert mb.beta_pdf(1.0, 1, 1) == 1.0
    
    def test_beta_pdf_invalid_input(self):
        """Test invalid inputs"""
        with pytest.raises(ValueError):
            mb.beta_pdf(-0.1, 2, 2)  # x out of range
        
        with pytest.raises(ValueError):
            mb.beta_pdf(1.1, 2, 2)   # x out of range
        
        with pytest.raises(ValueError):
            mb.beta_pdf(0.5, 0, 2)   # alpha <= 0
        
        with pytest.raises(ValueError):
            mb.beta_pdf(0.5, 2, -1)  # beta <= 0


class TestBetaCdf:
    """Test cases for beta_cdf"""
    
    def test_beta_cdf_uniform(self):
        """Test CDF for uniform distribution Beta(1,1)"""
        # For Beta(1,1), CDF(x) = x
        assert abs(mb.beta_cdf(0.0, 1, 1) - 0.0) < 1e-3
        assert abs(mb.beta_cdf(0.5, 1, 1) - 0.5) < 1e-3
        assert abs(mb.beta_cdf(1.0, 1, 1) - 1.0) < 1e-3
    
    def test_beta_cdf_symmetric(self):
        """Test CDF for symmetric distribution"""
        # Beta(2,2) should have CDF(0.5) = 0.5
        assert abs(mb.beta_cdf(0.5, 2, 2) - 0.5) < 1e-2
    
    def test_beta_cdf_monotonic(self):
        """Test that CDF is monotonically increasing"""
        x_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        cdf_values = [mb.beta_cdf(x, 2, 3) for x in x_values]
        
        # Check monotonicity
        for i in range(1, len(cdf_values)):
            assert cdf_values[i] >= cdf_values[i-1]
    
    def test_beta_cdf_boundary(self):
        """Test boundary values"""
        assert mb.beta_cdf(0.0, 2, 2) == 0.0
        assert mb.beta_cdf(1.0, 2, 2) == 1.0


class TestBetaMean:
    """Test cases for beta_mean"""
    
    def test_beta_mean_formula(self):
        """Test mean formula: α/(α+β)"""
        assert mb.beta_mean(1, 1) == 0.5
        assert mb.beta_mean(2, 2) == 0.5
        assert mb.beta_mean(1, 3) == 0.25
        assert mb.beta_mean(3, 1) == 0.75
    
    def test_beta_mean_invalid_input(self):
        """Test invalid inputs"""
        with pytest.raises(ValueError):
            mb.beta_mean(0, 2)
        
        with pytest.raises(ValueError):
            mb.beta_mean(2, -1)


class TestBetaVar:
    """Test cases for beta_var"""
    
    def test_beta_var_formula(self):
        """Test variance formula: αβ/((α+β)²(α+β+1))"""
        # Beta(1,1) variance = 1*1/(2²*3) = 1/12
        expected = 1/12
        assert abs(mb.beta_var(1, 1) - expected) < 1e-10
        
        # Beta(2,2) variance = 2*2/(4²*5) = 4/80 = 1/20
        expected = 1/20
        assert abs(mb.beta_var(2, 2) - expected) < 1e-10
    
    def test_beta_var_positive(self):
        """Test that variance is always positive"""
        test_cases = [(1, 1), (2, 3), (5, 2), (10, 10)]
        for alpha, beta in test_cases:
            assert mb.beta_var(alpha, beta) > 0


class TestBetaMode:
    """Test cases for beta_mode"""
    
    def test_beta_mode_formula(self):
        """Test mode formula: (α-1)/(α+β-2) for α,β > 1"""
        # Beta(2,2) mode = (2-1)/(2+2-2) = 1/2 = 0.5
        assert mb.beta_mode(2, 2) == 0.5
        
        # Beta(3,2) mode = (3-1)/(3+2-2) = 2/3
        expected = 2/3
        assert abs(mb.beta_mode(3, 2) - expected) < 1e-10
    
    def test_beta_mode_undefined(self):
        """Test cases where mode is undefined"""
        assert mb.beta_mode(1, 1) is None
        assert mb.beta_mode(1, 2) is None
        assert mb.beta_mode(2, 1) is None
        assert mb.beta_mode(0.5, 2) is None
    
    def test_beta_mode_valid_cases(self):
        """Test valid cases where mode exists"""
        assert mb.beta_mode(2, 2) is not None
        assert mb.beta_mode(3, 4) is not None
        assert mb.beta_mode(5, 2) is not None


if __name__ == "__main__":
    pytest.main([__file__])
