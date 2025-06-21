"""
Distance Measures Between Probability Distributions.

This module implements various distance and divergence measures
for comparing probability distributions.
"""

import math
from typing import List
from .utils import validate_distribution, safe_log


def hellinger_distance(p: List[float], q: List[float]) -> float:
    """
    Compute Hellinger distance between two probability distributions.
    
    H(p, q) = (1/√2) * √(∑(√p(x) - √q(x))²)
    
    Properties:
    - Bounded: H(p, q) ∈ [0, 1]
    - Symmetric: H(p, q) = H(q, p)
    - Metric: satisfies triangle inequality
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        Hellinger distance ∈ [0, 1]
        
    Raises:
        ValueError: If distributions are invalid or mismatched lengths
        
    Examples:
        >>> hellinger_distance([0.5, 0.5], [0.5, 0.5])  # Identical
        0.0
        >>> hellinger_distance([1.0, 0.0], [0.0, 1.0])  # Opposite
        1.0
        >>> hellinger_distance([0.8, 0.2], [0.6, 0.4])  # Moderate difference
        0.141
    """
    validate_distribution(p)
    validate_distribution(q)
    
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")
    
    sum_squared_diff = 0.0
    for p_i, q_i in zip(p, q):
        sqrt_p = math.sqrt(max(0, p_i))  # Ensure non-negative for sqrt
        sqrt_q = math.sqrt(max(0, q_i))
        sum_squared_diff += (sqrt_p - sqrt_q) ** 2
    
    return math.sqrt(sum_squared_diff) / math.sqrt(2)


def total_variation_distance(p: List[float], q: List[float]) -> float:
    """
    Compute total variation distance between two probability distributions.
    
    TV(p, q) = (1/2) * ∑|p(x) - q(x)|
    
    Properties:
    - Bounded: TV(p, q) ∈ [0, 1]
    - Symmetric: TV(p, q) = TV(q, p)
    - Metric: satisfies triangle inequality
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        Total variation distance ∈ [0, 1]
        
    Examples:
        >>> total_variation_distance([0.5, 0.5], [0.5, 0.5])  # Identical
        0.0
        >>> total_variation_distance([1.0, 0.0], [0.0, 1.0])  # Opposite
        1.0
        >>> total_variation_distance([0.7, 0.3], [0.4, 0.6])  # Moderate
        0.3
    """
    validate_distribution(p)
    validate_distribution(q)
    
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")
    
    total_diff = sum(abs(p_i - q_i) for p_i, q_i in zip(p, q))
    
    return total_diff / 2.0


def bhattacharyya_distance(p: List[float], q: List[float]) -> float:
    """
    Compute Bhattacharyya distance between two probability distributions.
    
    D_B(p, q) = -ln(∑√(p(x) * q(x)))
    
    Related to Hellinger distance: H²(p, q) = 1 - BC(p, q)
    where BC is the Bhattacharyya coefficient.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        Bhattacharyya distance ≥ 0
        
    Examples:
        >>> bhattacharyya_distance([0.5, 0.5], [0.5, 0.5])  # Identical
        0.0
        >>> bhattacharyya_distance([1.0, 0.0], [0.0, 1.0])  # No overlap
        inf
        >>> bhattacharyya_distance([0.8, 0.2], [0.6, 0.4])  # Partial overlap
        0.020
    """
    validate_distribution(p)
    validate_distribution(q)
    
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")
    
    # Bhattacharyya coefficient
    bc = 0.0
    for p_i, q_i in zip(p, q):
        if p_i > 0 and q_i > 0:
            bc += math.sqrt(p_i * q_i)
    
    if bc <= 0:
        return float('inf')  # No overlap
    
    return -math.log(bc)


def wasserstein_distance_1d(p: List[float], q: List[float], 
                           positions: List[float] = None) -> float:
    """
    Compute 1D Wasserstein distance (Earth Mover's Distance).
    
    For discrete distributions on ordered support, this is the
    area between cumulative distribution functions.
    
    W₁(p, q) = ∑|F_p(x) - F_q(x)|
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        positions: Positions of probability masses (default: 0, 1, 2, ...)
        
    Returns:
        Wasserstein distance ≥ 0
        
    Examples:
        >>> # Identical distributions
        >>> wasserstein_distance_1d([0.5, 0.5], [0.5, 0.5])
        0.0
        
        >>> # Shifted distributions
        >>> wasserstein_distance_1d([1.0, 0.0], [0.0, 1.0])
        1.0
        
        >>> # Custom positions
        >>> wasserstein_distance_1d([0.5, 0.5], [0.5, 0.5], [0, 10])
        0.0
    """
    validate_distribution(p)
    validate_distribution(q)
    
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")
    
    n = len(p)
    
    # Default positions: 0, 1, 2, ..., n-1
    if positions is None:
        positions = list(range(n))
    
    if len(positions) != n:
        raise ValueError("Positions must match distribution length")
    
    # Sort by positions to ensure proper ordering
    sorted_data = sorted(zip(positions, p, q))
    sorted_positions, sorted_p, sorted_q = zip(*sorted_data)
    
    # Compute cumulative distributions
    cum_p = 0.0
    cum_q = 0.0
    wasserstein = 0.0
    
    for i in range(n - 1):  # Don't include last point (both CDFs = 1)
        cum_p += sorted_p[i]
        cum_q += sorted_q[i]
        
        # Distance between positions
        dx = sorted_positions[i + 1] - sorted_positions[i]
        
        # Add area between CDFs
        wasserstein += abs(cum_p - cum_q) * dx
    
    return wasserstein
