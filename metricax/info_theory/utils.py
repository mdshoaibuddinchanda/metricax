"""
Utility functions for Information Theory computations.

This module provides helper functions for distribution validation,
normalization, and safe mathematical operations.
"""

import math
from typing import List, Tuple


def validate_distribution(p: List[float], tolerance: float = 1e-9) -> None:
    """
    Validate that a list represents a valid probability distribution.
    
    Args:
        p: List of probability values
        tolerance: Tolerance for sum check (default: 1e-9)
        
    Raises:
        ValueError: If p is not a valid probability distribution
        
    Examples:
        >>> validate_distribution([0.3, 0.7])  # Valid
        >>> validate_distribution([0.5, 0.6])  # Raises ValueError (sum > 1)
    """
    if not p:
        raise ValueError("Distribution cannot be empty")
    
    if not all(isinstance(x, (int, float)) for x in p):
        raise ValueError("All elements must be numeric")
    
    if not all(math.isfinite(x) for x in p):
        raise ValueError("All probabilities must be finite")
    
    if any(x < 0 for x in p):
        raise ValueError("All probabilities must be non-negative")
    
    total = sum(p)
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"Probabilities must sum to 1, got {total}")


def normalize_distribution(p: List[float]) -> List[float]:
    """
    Normalize a list of non-negative values to form a probability distribution.
    
    Args:
        p: List of non-negative values
        
    Returns:
        Normalized probability distribution that sums to 1
        
    Raises:
        ValueError: If input is invalid
        
    Examples:
        >>> normalize_distribution([1, 2, 3])
        [0.16666666666666666, 0.3333333333333333, 0.5]
        >>> normalize_distribution([0.2, 0.3, 0.1])
        [0.3333333333333333, 0.5, 0.16666666666666666]
    """
    if not p:
        raise ValueError("Cannot normalize empty list")
    
    if not all(isinstance(x, (int, float)) for x in p):
        raise ValueError("All elements must be numeric")
    
    if not all(math.isfinite(x) for x in p):
        raise ValueError("All values must be finite")
    
    if any(x < 0 for x in p):
        raise ValueError("All values must be non-negative")
    
    total = sum(p)
    if total <= 0:
        raise ValueError("Sum of values must be positive")
    
    return [x / total for x in p]


def joint_distribution(p_x: List[float], p_y_given_x: List[List[float]]) -> List[List[float]]:
    """
    Construct joint distribution from marginal and conditional distributions.
    
    P(X=i, Y=j) = P(X=i) * P(Y=j | X=i)
    
    Args:
        p_x: Marginal distribution P(X)
        p_y_given_x: Conditional distributions P(Y|X=i) for each i
        
    Returns:
        Joint distribution P(X, Y) as 2D list
        
    Raises:
        ValueError: If distributions are invalid or incompatible
        
    Examples:
        >>> p_x = [0.6, 0.4]
        >>> p_y_given_x = [[0.8, 0.2], [0.3, 0.7]]
        >>> joint = joint_distribution(p_x, p_y_given_x)
        >>> joint
        [[0.48, 0.12], [0.12, 0.28]]
    """
    validate_distribution(p_x)
    
    if len(p_y_given_x) != len(p_x):
        raise ValueError("Number of conditional distributions must match marginal size")
    
    # Validate each conditional distribution
    for i, p_y_i in enumerate(p_y_given_x):
        try:
            validate_distribution(p_y_i)
        except ValueError as e:
            raise ValueError(f"Invalid conditional distribution P(Y|X={i}): {e}")
    
    # Check that all conditional distributions have same size
    y_size = len(p_y_given_x[0])
    if not all(len(p_y_i) == y_size for p_y_i in p_y_given_x):
        raise ValueError("All conditional distributions must have same size")
    
    # Construct joint distribution
    joint = []
    for i, (p_x_i, p_y_i) in enumerate(zip(p_x, p_y_given_x)):
        joint_row = [p_x_i * p_y_j for p_y_j in p_y_i]
        joint.append(joint_row)
    
    return joint


def safe_log(x: float, base: float = 2.0, epsilon: float = 1e-15) -> float:
    """
    Compute logarithm with numerical safety for small values.
    
    Args:
        x: Input value
        base: Logarithm base
        epsilon: Small value to add for numerical stability
        
    Returns:
        log_base(max(x, epsilon))
        
    Examples:
        >>> safe_log(0.5)
        -1.0
        >>> safe_log(0.0)  # Returns log(epsilon) instead of -inf
        -49.82892142331044
    """
    if x <= 0:
        x = epsilon
    
    if base == math.e:
        return math.log(x)
    elif base == 2.0:
        return math.log2(x)
    elif base == 10.0:
        return math.log10(x)
    else:
        return math.log(x) / math.log(base)
