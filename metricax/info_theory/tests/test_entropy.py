"""
Tests for Information Theory Entropy Functions

Comprehensive test suite covering:
- Shannon entropy
- Cross-entropy  
- KL divergence
- JS divergence
- Renyi entropy
- Tsallis entropy

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


class TestEntropy:
    """Test Shannon entropy function."""
    
    def test_uniform_distribution(self):
        """Test entropy of uniform distributions."""
        # Binary uniform
        assert abs(it.entropy([0.5, 0.5]) - 1.0) < TEST_PRECISION
        
        # 4-way uniform
        expected = math.log2(4)
        assert abs(it.entropy([0.25, 0.25, 0.25, 0.25]) - expected) < TEST_PRECISION
        
        # 8-way uniform
        expected = math.log2(8)
        uniform_8 = [0.125] * 8
        assert abs(it.entropy(uniform_8) - expected) < TEST_PRECISION
    
    def test_deterministic_distribution(self):
        """Test entropy of deterministic distributions."""
        # Single outcome
        assert abs(it.entropy([1.0, 0.0]) - 0.0) < TEST_PRECISION
        assert abs(it.entropy([0.0, 1.0]) - 0.0) < TEST_PRECISION
        assert abs(it.entropy([1.0, 0.0, 0.0, 0.0]) - 0.0) < TEST_PRECISION
    
    def test_known_values(self):
        """Test entropy for distributions with known analytical values."""
        # Binary distribution with known entropy
        p = 0.3
        expected = -p * math.log2(p) - (1-p) * math.log2(1-p)
        assert abs(it.entropy([p, 1-p]) - expected) < TEST_PRECISION
        
        # Three-way distribution
        dist = [0.5, 0.3, 0.2]
        expected = sum(-p * math.log2(p) for p in dist if p > 0)
        assert abs(it.entropy(dist) - expected) < TEST_PRECISION
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty distribution
        with pytest.raises(ValueError):
            it.entropy([])
        
        # Negative probabilities
        with pytest.raises(ValueError):
            it.entropy([-0.1, 1.1])
        
        # Probabilities don't sum to 1
        with pytest.raises(ValueError):
            it.entropy([0.3, 0.3])  # Sum = 0.6
        
        # Zero probabilities (should be handled gracefully)
        assert abs(it.entropy([0.0, 1.0]) - 0.0) < TEST_PRECISION
        assert abs(it.entropy([0.5, 0.0, 0.5]) - 1.0) < TEST_PRECISION


class TestCrossEntropy:
    """Test cross-entropy function."""
    
    def test_identical_distributions(self):
        """Test cross-entropy when distributions are identical."""
        dist = [0.3, 0.4, 0.3]
        # H(p, p) = H(p)
        cross_ent = it.cross_entropy(dist, dist)
        entropy = it.entropy(dist)
        assert abs(cross_ent - entropy) < TEST_PRECISION
    
    def test_known_values(self):
        """Test cross-entropy for known analytical values."""
        p = [0.5, 0.5]
        q = [0.3, 0.7]
        
        # H(p, q) = -∑ p(x) log q(x)
        expected = -0.5 * math.log2(0.3) - 0.5 * math.log2(0.7)
        assert abs(it.cross_entropy(p, q) - expected) < TEST_PRECISION
    
    def test_asymmetry(self):
        """Test that cross-entropy is asymmetric."""
        p = [0.7, 0.3]
        q = [0.3, 0.7]
        
        h_pq = it.cross_entropy(p, q)
        h_qp = it.cross_entropy(q, p)
        
        # Should be different (asymmetric)
        assert abs(h_pq - h_qp) > TEST_PRECISION
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Mismatched lengths
        with pytest.raises(ValueError):
            it.cross_entropy([0.5, 0.5], [0.3, 0.3, 0.4])
        
        # Zero in q where p > 0 (should handle gracefully or raise error)
        with pytest.raises((ValueError, ZeroDivisionError)):
            it.cross_entropy([0.5, 0.5], [0.0, 1.0])


class TestKLDivergence:
    """Test KL divergence function."""
    
    def test_identical_distributions(self):
        """Test KL divergence when distributions are identical."""
        dist = [0.3, 0.4, 0.3]
        # D(p || p) = 0
        assert abs(it.kl_divergence(dist, dist) - 0.0) < TEST_PRECISION
    
    def test_known_values(self):
        """Test KL divergence for known analytical values."""
        p = [0.5, 0.5]
        q = [0.25, 0.75]
        
        # D(p || q) = ∑ p(x) log(p(x) / q(x))
        expected = 0.5 * math.log2(0.5 / 0.25) + 0.5 * math.log2(0.5 / 0.75)
        assert abs(it.kl_divergence(p, q) - expected) < TEST_PRECISION
    
    def test_asymmetry(self):
        """Test that KL divergence is asymmetric."""
        p = [0.7, 0.3]
        q = [0.3, 0.7]
        
        d_pq = it.kl_divergence(p, q)
        d_qp = it.kl_divergence(q, p)
        
        # Should be different (asymmetric)
        assert abs(d_pq - d_qp) > TEST_PRECISION
    
    def test_non_negativity(self):
        """Test that KL divergence is always non-negative."""
        test_cases = [
            ([0.5, 0.5], [0.3, 0.7]),
            ([0.8, 0.2], [0.2, 0.8]),
            ([0.33, 0.33, 0.34], [0.25, 0.25, 0.5]),
        ]
        
        for p, q in test_cases:
            assert it.kl_divergence(p, q) >= -TEST_PRECISION  # Allow small numerical errors


class TestJSDivergence:
    """Test Jensen-Shannon divergence function."""
    
    def test_identical_distributions(self):
        """Test JS divergence when distributions are identical."""
        dist = [0.3, 0.4, 0.3]
        # JS(p, p) = 0
        assert abs(it.js_divergence(dist, dist) - 0.0) < TEST_PRECISION
    
    def test_symmetry(self):
        """Test that JS divergence is symmetric."""
        p = [0.7, 0.3]
        q = [0.3, 0.7]
        
        js_pq = it.js_divergence(p, q)
        js_qp = it.js_divergence(q, p)
        
        # Should be equal (symmetric)
        assert abs(js_pq - js_qp) < TEST_PRECISION
    
    def test_bounded(self):
        """Test that JS divergence is bounded by log(2)."""
        test_cases = [
            ([1.0, 0.0], [0.0, 1.0]),  # Maximum divergence case
            ([0.5, 0.5], [0.3, 0.7]),
            ([0.8, 0.2], [0.2, 0.8]),
        ]
        
        for p, q in test_cases:
            js_div = it.js_divergence(p, q)
            assert 0 <= js_div <= math.log2(2) + TEST_PRECISION
    
    def test_maximum_divergence(self):
        """Test JS divergence for maximum divergence case."""
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        
        # Should be close to log(2) for maximum divergence
        js_div = it.js_divergence(p, q)
        assert abs(js_div - math.log2(2)) < TEST_PRECISION


class TestRenyiEntropy:
    """Test Renyi entropy function."""
    
    def test_alpha_one_equals_shannon(self):
        """Test that Renyi entropy with α=1 equals Shannon entropy."""
        dist = [0.3, 0.4, 0.3]
        
        # Limit as α → 1 should equal Shannon entropy
        renyi_close_to_one = it.renyi_entropy(dist, 1.001)
        shannon = it.entropy(dist)
        
        # Should be very close
        assert abs(renyi_close_to_one - shannon) < 0.01
    
    def test_alpha_zero(self):
        """Test Renyi entropy with α=0 (log of support size)."""
        dist = [0.3, 0.4, 0.3]
        expected = math.log2(3)  # log of number of non-zero elements
        
        assert abs(it.renyi_entropy(dist, 0) - expected) < TEST_PRECISION
    
    def test_alpha_infinity(self):
        """Test Renyi entropy with α=∞ (min-entropy)."""
        dist = [0.5, 0.3, 0.2]
        expected = -math.log2(0.5)  # -log of maximum probability
        
        # Use large alpha to approximate infinity
        renyi_large_alpha = it.renyi_entropy(dist, 100)
        assert abs(renyi_large_alpha - expected) < 0.01
    
    def test_monotonicity(self):
        """Test that Renyi entropy is monotonic in α."""
        dist = [0.5, 0.3, 0.2]
        
        # For α < β, H_α(X) ≥ H_β(X)
        h_05 = it.renyi_entropy(dist, 0.5)
        h_2 = it.renyi_entropy(dist, 2.0)
        h_5 = it.renyi_entropy(dist, 5.0)
        
        assert h_05 >= h_2 - TEST_PRECISION
        assert h_2 >= h_5 - TEST_PRECISION


class TestTsallisEntropy:
    """Test Tsallis entropy function."""
    
    def test_q_one_limit(self):
        """Test that Tsallis entropy approaches Shannon entropy as q→1."""
        dist = [0.3, 0.4, 0.3]
        
        # Limit as q → 1 should equal Shannon entropy
        tsallis_close_to_one = it.tsallis_entropy(dist, 1.001)
        shannon = it.entropy(dist)
        
        # Should be very close
        assert abs(tsallis_close_to_one - shannon) < 0.01
    
    def test_q_zero(self):
        """Test Tsallis entropy with q=0."""
        dist = [0.3, 0.4, 0.3]
        expected = len([p for p in dist if p > 0]) - 1  # Support size - 1
        
        assert abs(it.tsallis_entropy(dist, 0) - expected) < TEST_PRECISION
    
    def test_q_two(self):
        """Test Tsallis entropy with q=2 (collision entropy)."""
        dist = [0.5, 0.3, 0.2]
        
        # S_2(X) = 1 - ∑ p_i^2
        expected = 1 - sum(p**2 for p in dist)
        
        assert abs(it.tsallis_entropy(dist, 2) - expected) < TEST_PRECISION
    
    def test_non_additivity(self):
        """Test non-additive property of Tsallis entropy."""
        # For independent systems, Tsallis entropy is non-additive
        # This is a characteristic property that distinguishes it from Shannon entropy
        
        p1 = [0.6, 0.4]
        p2 = [0.7, 0.3]
        q = 2.0
        
        # Individual entropies
        s1 = it.tsallis_entropy(p1, q)
        s2 = it.tsallis_entropy(p2, q)
        
        # Joint distribution (independent)
        joint = [p1[i] * p2[j] for i in range(2) for j in range(2)]
        s_joint = it.tsallis_entropy(joint, q)
        
        # For Tsallis: S_q(A+B) = S_q(A) + S_q(B) + (1-q)S_q(A)S_q(B)
        expected_joint = s1 + s2 + (1-q) * s1 * s2
        
        assert abs(s_joint - expected_joint) < TEST_PRECISION


if __name__ == "__main__":
    pytest.main([__file__])
