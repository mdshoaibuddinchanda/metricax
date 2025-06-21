"""
Tests for Information Theory Distance Measures

Comprehensive test suite covering:
- Hellinger distance
- Total variation distance
- Bhattacharyya distance
- Wasserstein distance (1D)

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


class TestHellingerDistance:
    """Test Hellinger distance function."""
    
    def test_identical_distributions(self):
        """Test Hellinger distance for identical distributions (should be 0)."""
        dist = [0.3, 0.4, 0.3]
        hd = it.hellinger_distance(dist, dist)
        assert abs(hd - 0.0) < TEST_PRECISION
    
    def test_maximum_distance(self):
        """Test Hellinger distance for maximally different distributions."""
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        
        hd = it.hellinger_distance(p, q)
        expected = 1.0  # Maximum Hellinger distance
        assert abs(hd - expected) < TEST_PRECISION
    
    def test_known_values(self):
        """Test Hellinger distance for known analytical values."""
        p = [0.6, 0.4]
        q = [0.4, 0.6]
        
        # H(p,q) = (1/√2) * √(∑(√p(x) - √q(x))²)
        expected = (1/math.sqrt(2)) * math.sqrt(
            (math.sqrt(0.6) - math.sqrt(0.4))**2 + 
            (math.sqrt(0.4) - math.sqrt(0.6))**2
        )
        
        hd = it.hellinger_distance(p, q)
        assert abs(hd - expected) < TEST_PRECISION
    
    def test_symmetry(self):
        """Test that Hellinger distance is symmetric."""
        p = [0.7, 0.2, 0.1]
        q = [0.3, 0.5, 0.2]
        
        hd_pq = it.hellinger_distance(p, q)
        hd_qp = it.hellinger_distance(q, p)
        
        assert abs(hd_pq - hd_qp) < TEST_PRECISION
    
    def test_bounded_range(self):
        """Test that Hellinger distance is bounded in [0, 1]."""
        test_cases = [
            ([0.5, 0.5], [0.3, 0.7]),
            ([0.8, 0.1, 0.1], [0.1, 0.8, 0.1]),
            ([0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1]),
        ]
        
        for p, q in test_cases:
            hd = it.hellinger_distance(p, q)
            assert 0.0 <= hd <= 1.0 + TEST_PRECISION
    
    def test_triangle_inequality(self):
        """Test triangle inequality property."""
        p = [0.5, 0.3, 0.2]
        q = [0.3, 0.4, 0.3]
        r = [0.2, 0.2, 0.6]
        
        hd_pq = it.hellinger_distance(p, q)
        hd_qr = it.hellinger_distance(q, r)
        hd_pr = it.hellinger_distance(p, r)
        
        # Triangle inequality: d(p,r) ≤ d(p,q) + d(q,r)
        assert hd_pr <= hd_pq + hd_qr + TEST_PRECISION
    
    def test_input_validation(self):
        """Test error handling for invalid inputs."""
        # Mismatched lengths
        with pytest.raises(ValueError):
            it.hellinger_distance([0.5, 0.5], [0.3, 0.3, 0.4])
        
        # Invalid distributions
        with pytest.raises(ValueError):
            it.hellinger_distance([0.3, 0.3], [0.5, 0.5])  # Don't sum to 1


class TestTotalVariationDistance:
    """Test total variation distance function."""
    
    def test_identical_distributions(self):
        """Test TV distance for identical distributions (should be 0)."""
        dist = [0.4, 0.3, 0.3]
        tvd = it.total_variation_distance(dist, dist)
        assert abs(tvd - 0.0) < TEST_PRECISION
    
    def test_maximum_distance(self):
        """Test TV distance for maximally different distributions."""
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        
        tvd = it.total_variation_distance(p, q)
        expected = 1.0  # Maximum TV distance
        assert abs(tvd - expected) < TEST_PRECISION
    
    def test_known_values(self):
        """Test TV distance for known analytical values."""
        p = [0.7, 0.3]
        q = [0.4, 0.6]
        
        # TV(p,q) = (1/2) * ∑|p(x) - q(x)|
        expected = 0.5 * (abs(0.7 - 0.4) + abs(0.3 - 0.6))
        
        tvd = it.total_variation_distance(p, q)
        assert abs(tvd - expected) < TEST_PRECISION
    
    def test_symmetry(self):
        """Test that TV distance is symmetric."""
        p = [0.6, 0.2, 0.2]
        q = [0.2, 0.6, 0.2]
        
        tvd_pq = it.total_variation_distance(p, q)
        tvd_qp = it.total_variation_distance(q, p)
        
        assert abs(tvd_pq - tvd_qp) < TEST_PRECISION
    
    def test_bounded_range(self):
        """Test that TV distance is bounded in [0, 1]."""
        test_cases = [
            ([0.5, 0.5], [0.3, 0.7]),
            ([0.9, 0.05, 0.05], [0.1, 0.45, 0.45]),
            ([0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]),
        ]
        
        for p, q in test_cases:
            tvd = it.total_variation_distance(p, q)
            assert 0.0 <= tvd <= 1.0 + TEST_PRECISION
    
    def test_triangle_inequality(self):
        """Test triangle inequality property."""
        p = [0.4, 0.3, 0.3]
        q = [0.3, 0.4, 0.3]
        r = [0.2, 0.3, 0.5]
        
        tvd_pq = it.total_variation_distance(p, q)
        tvd_qr = it.total_variation_distance(q, r)
        tvd_pr = it.total_variation_distance(p, r)
        
        # Triangle inequality: d(p,r) ≤ d(p,q) + d(q,r)
        assert tvd_pr <= tvd_pq + tvd_qr + TEST_PRECISION


class TestBhattacharyyaDistance:
    """Test Bhattacharyya distance function."""
    
    def test_identical_distributions(self):
        """Test Bhattacharyya distance for identical distributions (should be 0)."""
        dist = [0.3, 0.5, 0.2]
        bd = it.bhattacharyya_distance(dist, dist)
        assert abs(bd - 0.0) < TEST_PRECISION
    
    def test_no_overlap(self):
        """Test Bhattacharyya distance for non-overlapping distributions."""
        p = [1.0, 0.0, 0.0]
        q = [0.0, 1.0, 0.0]
        
        bd = it.bhattacharyya_distance(p, q)
        assert bd == float('inf')  # No overlap
    
    def test_known_values(self):
        """Test Bhattacharyya distance for known analytical values."""
        p = [0.5, 0.5]
        q = [0.3, 0.7]
        
        # BC(p,q) = ∑√(p(x) * q(x))
        bc = math.sqrt(0.5 * 0.3) + math.sqrt(0.5 * 0.7)
        expected = -math.log(bc)
        
        bd = it.bhattacharyya_distance(p, q)
        assert abs(bd - expected) < TEST_PRECISION
    
    def test_symmetry(self):
        """Test that Bhattacharyya distance is symmetric."""
        p = [0.6, 0.3, 0.1]
        q = [0.2, 0.5, 0.3]
        
        bd_pq = it.bhattacharyya_distance(p, q)
        bd_qp = it.bhattacharyya_distance(q, p)
        
        assert abs(bd_pq - bd_qp) < TEST_PRECISION
    
    def test_non_negativity(self):
        """Test that Bhattacharyya distance is non-negative."""
        test_cases = [
            ([0.5, 0.5], [0.4, 0.6]),
            ([0.7, 0.2, 0.1], [0.3, 0.4, 0.3]),
            ([0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1]),
        ]
        
        for p, q in test_cases:
            bd = it.bhattacharyya_distance(p, q)
            assert bd >= -TEST_PRECISION
    
    def test_relationship_to_hellinger(self):
        """Test relationship between Bhattacharyya and Hellinger distances."""
        p = [0.6, 0.4]
        q = [0.3, 0.7]
        
        bd = it.bhattacharyya_distance(p, q)
        hd = it.hellinger_distance(p, q)
        
        # Relationship: H²(p,q) = 2(1 - e^(-BD(p,q)))
        expected_hd_squared = 2 * (1 - math.exp(-bd))
        actual_hd_squared = hd ** 2
        
        assert abs(actual_hd_squared - expected_hd_squared) < TEST_PRECISION


class TestWassersteinDistance1D:
    """Test 1D Wasserstein distance function."""
    
    def test_identical_distributions(self):
        """Test Wasserstein distance for identical distributions (should be 0)."""
        dist = [0.3, 0.4, 0.3]
        wd = it.wasserstein_distance_1d(dist, dist)
        assert abs(wd - 0.0) < TEST_PRECISION
    
    def test_shifted_distributions(self):
        """Test Wasserstein distance for shifted distributions."""
        p = [1.0, 0.0, 0.0]  # Mass at position 0
        q = [0.0, 1.0, 0.0]  # Mass at position 1
        
        wd = it.wasserstein_distance_1d(p, q)
        expected = 1.0  # Distance between positions 0 and 1
        assert abs(wd - expected) < TEST_PRECISION
    
    def test_custom_positions(self):
        """Test Wasserstein distance with custom positions."""
        p = [0.5, 0.5]
        q = [0.5, 0.5]
        positions = [0, 10]  # Positions 0 and 10
        
        wd = it.wasserstein_distance_1d(p, q, positions)
        assert abs(wd - 0.0) < TEST_PRECISION  # Identical distributions
        
        # Different distributions
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        wd = it.wasserstein_distance_1d(p, q, positions)
        expected = 10.0  # Distance between positions 0 and 10
        assert abs(wd - expected) < TEST_PRECISION
    
    def test_symmetry(self):
        """Test that Wasserstein distance is symmetric."""
        p = [0.6, 0.3, 0.1]
        q = [0.2, 0.5, 0.3]
        
        wd_pq = it.wasserstein_distance_1d(p, q)
        wd_qp = it.wasserstein_distance_1d(q, p)
        
        assert abs(wd_pq - wd_qp) < TEST_PRECISION
    
    def test_non_negativity(self):
        """Test that Wasserstein distance is non-negative."""
        test_cases = [
            ([0.5, 0.5], [0.3, 0.7]),
            ([0.8, 0.1, 0.1], [0.1, 0.8, 0.1]),
            ([0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1]),
        ]
        
        for p, q in test_cases:
            wd = it.wasserstein_distance_1d(p, q)
            assert wd >= -TEST_PRECISION
    
    def test_triangle_inequality(self):
        """Test triangle inequality property."""
        p = [0.5, 0.3, 0.2]
        q = [0.3, 0.4, 0.3]
        r = [0.2, 0.2, 0.6]
        
        wd_pq = it.wasserstein_distance_1d(p, q)
        wd_qr = it.wasserstein_distance_1d(q, r)
        wd_pr = it.wasserstein_distance_1d(p, r)
        
        # Triangle inequality: d(p,r) ≤ d(p,q) + d(q,r)
        assert wd_pr <= wd_pq + wd_qr + TEST_PRECISION
    
    def test_input_validation(self):
        """Test error handling for invalid inputs."""
        # Mismatched lengths
        with pytest.raises(ValueError):
            it.wasserstein_distance_1d([0.5, 0.5], [0.3, 0.3, 0.4])
        
        # Mismatched positions length
        with pytest.raises(ValueError):
            it.wasserstein_distance_1d([0.5, 0.5], [0.3, 0.7], [0, 1, 2])
        
        # Invalid distributions
        with pytest.raises(ValueError):
            it.wasserstein_distance_1d([0.3, 0.3], [0.5, 0.5])  # Don't sum to 1


if __name__ == "__main__":
    pytest.main([__file__])
