"""
Code Length and Optimal Coding Theory.

This module implements functions related to optimal coding,
code length bounds, and information-theoretic limits.
"""

import math
from typing import List
from .utils import validate_distribution, safe_log
from .entropy import entropy


def optimal_code_length(p: List[float], base: float = 2.0) -> List[float]:
    """
    Compute optimal code length for each symbol (Shannon code).
    
    L*(x) = -log_base(p(x))
    
    The expected code length equals the entropy H(X).
    
    Args:
        p: Probability distribution
        base: Logarithm base (2 for binary codes)
        
    Returns:
        List of optimal code lengths for each symbol
        
    Raises:
        ValueError: If distribution is invalid
        
    Examples:
        >>> optimal_code_length([0.5, 0.5])
        [1.0, 1.0]
        >>> optimal_code_length([0.25, 0.75])
        [2.0, 0.415]
        >>> optimal_code_length([0.8, 0.1, 0.1])
        [0.322, 3.322, 3.322]
    """
    validate_distribution(p)
    
    if base <= 0 or base == 1:
        raise ValueError("Base must be positive and not equal to 1")
    
    code_lengths = []
    for prob in p:
        if prob <= 0:
            code_lengths.append(float('inf'))  # Infinite code length for impossible events
        else:
            code_lengths.append(-safe_log(prob, base))
    
    return code_lengths


def fano_inequality(conditional_entropy: float, alphabet_size: int, base: float = 2.0) -> float:
    """
    Compute Fano's inequality lower bound on error probability.

    Pe ≥ (H(X|Y) - 1) / log(|X| - 1)

    where Pe is the error probability and |X| is alphabet size.

    Args:
        conditional_entropy: Conditional entropy H(X|Y)
        alphabet_size: Size of alphabet |X| ≥ 2
        base: Logarithm base

    Returns:
        Lower bound on error probability

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> # Perfect prediction (H(X|Y) = 0) implies Pe = 0
        >>> fano_inequality(0.0, 2)
        0.0

        >>> # High conditional entropy implies high error probability
        >>> fano_inequality(1.5, 4)
        0.5
    """
    if alphabet_size < 2:
        raise ValueError("Alphabet size must be at least 2")
    if conditional_entropy < 0:
        raise ValueError("Conditional entropy must be non-negative")

    if conditional_entropy == 0:
        return 0.0

    # Fano bound: Pe ≥ (H(X|Y) - 1) / log(|X| - 1)
    if alphabet_size == 2:
        # Special case for binary alphabet
        return max(0.0, conditional_entropy - 1.0)

    denominator = safe_log(alphabet_size - 1, base)
    if denominator == 0:
        return 0.0

    fano_bound = max(0.0, (conditional_entropy - 1.0) / denominator)

    return min(1.0, fano_bound)  # Error probability cannot exceed 1


def redundancy(p: List[float], code_lengths: List[float], base: float = 2.0) -> float:
    """
    Compute redundancy of a code over the optimal (entropy) limit.
    
    Redundancy = L̄ - H(X)
    where L̄ = ∑ p(x) * L(x) is expected code length
    
    Args:
        p: Probability distribution
        code_lengths: Actual code lengths for each symbol
        base: Logarithm base
        
    Returns:
        Redundancy ≥ 0 (0 for optimal codes)
        
    Raises:
        ValueError: If distributions are invalid or mismatched lengths
        
    Examples:
        >>> # Optimal code has zero redundancy
        >>> p = [0.5, 0.5]
        >>> optimal_lengths = optimal_code_length(p)
        >>> redundancy(p, optimal_lengths)
        0.0
        
        >>> # Suboptimal code has positive redundancy
        >>> redundancy([0.8, 0.2], [1, 1])  # Fixed-length code
        0.278
    """
    validate_distribution(p)
    
    if len(p) != len(code_lengths):
        raise ValueError("Distribution and code lengths must have same size")
    
    if any(length < 0 for length in code_lengths):
        raise ValueError("Code lengths must be non-negative")
    
    # Expected code length
    expected_length = sum(prob * length for prob, length in zip(p, code_lengths))
    
    # Entropy (optimal expected code length)
    h_x = entropy(p, base)
    
    return max(0.0, expected_length - h_x)
