"""
Unit tests for metricax.bayesian.utils module.
"""

import pytest
import math
import sys
import os

# Add the parent directory to the path to import metricax
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import metricax.bayesian as mb


class TestGammaFunc:
    """Test cases for gamma_func"""
    
    def test_gamma_func_positive_integers(self):
        """Test gamma function for positive integers"""
        assert mb.gamma_func(1.0) == 1.0
        assert mb.gamma_func(2.0) == 1.0
        assert mb.gamma_func(3.0) == 2.0
        assert mb.gamma_func(4.0) == 6.0
    
    def test_gamma_func_positive_floats(self):
        """Test gamma function for positive floats"""
        result = mb.gamma_func(2.5)
        expected = math.gamma(2.5)
        assert abs(result - expected) < 1e-10
    
    def test_gamma_func_invalid_input(self):
        """Test gamma function with invalid inputs"""
        with pytest.raises(ValueError):
            mb.gamma_func(0.0)
        
        with pytest.raises(ValueError):
            mb.gamma_func(-1.0)
        
        with pytest.raises(ValueError):
            mb.gamma_func(float('inf'))
        
        with pytest.raises(ValueError):
            mb.gamma_func(float('nan'))


class TestValidateProb:
    """Test cases for validate_prob"""
    
    def test_valid_probabilities(self):
        """Test valid probability values"""
        assert mb.validate_prob(0.0) == True
        assert mb.validate_prob(0.5) == True
        assert mb.validate_prob(1.0) == True
        assert mb.validate_prob(0.001) == True
        assert mb.validate_prob(0.999) == True
    
    def test_invalid_probabilities(self):
        """Test invalid probability values"""
        assert mb.validate_prob(-0.1) == False
        assert mb.validate_prob(1.1) == False
        assert mb.validate_prob(float('inf')) == False
        assert mb.validate_prob(float('nan')) == False
        assert mb.validate_prob(-float('inf')) == False


class TestNormalize:
    """Test cases for normalize"""
    
    def test_normalize_basic(self):
        """Test basic normalization"""
        result = mb.normalize([1, 2, 3])
        expected = [1/6, 2/6, 3/6]
        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-10
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-10
    
    def test_normalize_already_normalized(self):
        """Test normalizing already normalized probabilities"""
        input_probs = [0.2, 0.3, 0.5]
        result = mb.normalize(input_probs)
        assert abs(sum(result) - 1.0) < 1e-10
        for r, i in zip(result, input_probs):
            assert abs(r - i) < 1e-10
    
    def test_normalize_single_element(self):
        """Test normalizing single element"""
        result = mb.normalize([5.0])
        assert result == [1.0]
    
    def test_normalize_invalid_input(self):
        """Test normalize with invalid inputs"""
        with pytest.raises(ValueError):
            mb.normalize([])
        
        with pytest.raises(ValueError):
            mb.normalize([0, 0, 0])
        
        with pytest.raises(ValueError):
            mb.normalize([-1, 2, 3])
        
        with pytest.raises(ValueError):
            mb.normalize([float('nan'), 1, 2])


class TestSafeDiv:
    """Test cases for safe_div"""
    
    def test_safe_div_normal(self):
        """Test normal division"""
        assert mb.safe_div(10, 2) == 5.0
        assert mb.safe_div(7, 3) == 7/3
        assert mb.safe_div(0, 5) == 0.0
    
    def test_safe_div_by_zero(self):
        """Test division by zero"""
        assert mb.safe_div(10, 0) == 0.0  # default
        assert mb.safe_div(10, 0, float('inf')) == float('inf')
        assert mb.safe_div(10, 0, -1) == -1
    
    def test_safe_div_infinite_inputs(self):
        """Test with infinite inputs"""
        result = mb.safe_div(float('inf'), 2)
        assert math.isnan(result)
        
        result = mb.safe_div(5, float('inf'))
        assert math.isnan(result)
        
        result = mb.safe_div(float('nan'), 2)
        assert math.isnan(result)


if __name__ == "__main__":
    pytest.main([__file__])
