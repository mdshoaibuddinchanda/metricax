"""
Utility functions for Bayesian statistics computations.

This module provides helper functions for mathematical operations,
input validation, and numerical stability in Bayesian computations.
"""

import math
from typing import List, Union


def gamma_func(x: float) -> float:
    """
    Compute the gamma function Γ(x).
    
    This is a wrapper around math.gamma with input validation.
    
    Args:
        x: Input value for gamma function
        
    Returns:
        Gamma function value Γ(x)
        
    Raises:
        ValueError: If x <= 0 or x is not finite
        
    Examples:
        >>> gamma_func(1.0)
        1.0
        >>> gamma_func(2.0)
        1.0
        >>> gamma_func(3.0)
        2.0
    """
    if not math.isfinite(x):
        raise ValueError("Input must be finite")
    if x <= 0:
        raise ValueError("Gamma function undefined for x <= 0")
    
    return math.gamma(x)


def validate_prob(x: float) -> bool:
    """
    Validate that a value is a valid probability (between 0 and 1).
    
    Args:
        x: Value to validate
        
    Returns:
        True if x is a valid probability, False otherwise
        
    Examples:
        >>> validate_prob(0.5)
        True
        >>> validate_prob(-0.1)
        False
        >>> validate_prob(1.5)
        False
    """
    return math.isfinite(x) and 0.0 <= x <= 1.0


def normalize(probs: List[float]) -> List[float]:
    """
    Normalize a list of probabilities to sum to 1.
    
    Args:
        probs: List of probability values
        
    Returns:
        Normalized probability list that sums to 1
        
    Raises:
        ValueError: If input is empty or all values are zero/negative
        
    Examples:
        >>> normalize([1, 2, 3])
        [0.16666666666666666, 0.3333333333333333, 0.5]
        >>> normalize([0.1, 0.2, 0.3])
        [0.16666666666666666, 0.3333333333333333, 0.5]
    """
    if not probs:
        raise ValueError("Cannot normalize empty list")
    
    # Check for valid inputs
    if not all(math.isfinite(p) and p >= 0 for p in probs):
        raise ValueError("All probabilities must be finite and non-negative")
    
    total = sum(probs)
    if total <= 0:
        raise ValueError("Sum of probabilities must be positive")
    
    return [p / total for p in probs]


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """
    Perform safe division with handling for division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        default: Value to return when b is zero (default: 0.0)
        
    Returns:
        a/b if b != 0, otherwise default value
        
    Examples:
        >>> safe_div(10, 2)
        5.0
        >>> safe_div(10, 0)
        0.0
        >>> safe_div(10, 0, float('inf'))
        inf
    """
    if not math.isfinite(a) or not math.isfinite(b):
        return float('nan')
    
    if b == 0:
        return default
    
    return a / b
