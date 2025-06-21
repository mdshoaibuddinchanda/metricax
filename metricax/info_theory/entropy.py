"""
Entropy and Variants for Information Theory.

This module implements core entropy measures including Shannon entropy,
cross-entropy, KL divergence, and their generalizations.
"""

import math
from typing import List, Union
from .utils import validate_distribution, safe_log


def entropy(p: List[float], base: float = 2.0) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    
    H(X) = -∑ p(x) * log_base(p(x))
    
    Args:
        p: Probability distribution (must sum to 1)
        base: Logarithm base (2 for bits, e for nats, 10 for dits)
        
    Returns:
        Shannon entropy in specified units
        
    Raises:
        ValueError: If p is not a valid probability distribution
        
    Examples:
        >>> entropy([0.5, 0.5])  # Maximum entropy for 2 outcomes
        1.0
        >>> entropy([1.0, 0.0])  # Minimum entropy (deterministic)
        0.0
        >>> entropy([0.25, 0.25, 0.25, 0.25])  # Uniform distribution
        2.0
    """
    validate_distribution(p)
    
    if base <= 0 or base == 1:
        raise ValueError("Base must be positive and not equal to 1")
    
    h = 0.0
    for prob in p:
        if prob > 0:  # Skip zero probabilities (0 * log(0) = 0)
            h -= prob * safe_log(prob, base)
    
    return h


def cross_entropy(p: List[float], q: List[float], base: float = 2.0) -> float:
    """
    Compute cross-entropy between two probability distributions.
    
    H(p, q) = -∑ p(x) * log_base(q(x))
    
    Args:
        p: True probability distribution
        q: Predicted/model probability distribution
        base: Logarithm base (2 for bits, e for nats)
        
    Returns:
        Cross-entropy between p and q
        
    Raises:
        ValueError: If distributions are invalid or mismatched lengths
        
    Examples:
        >>> cross_entropy([0.5, 0.5], [0.5, 0.5])  # Same distributions
        1.0
        >>> cross_entropy([1.0, 0.0], [0.9, 0.1])  # Close to true
        0.152
        >>> cross_entropy([1.0, 0.0], [0.1, 0.9])  # Far from true
        3.322
    """
    validate_distribution(p)
    validate_distribution(q)
    
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")
    
    if base <= 0 or base == 1:
        raise ValueError("Base must be positive and not equal to 1")
    
    ce = 0.0
    for p_i, q_i in zip(p, q):
        if p_i > 0:  # Only consider non-zero true probabilities
            if q_i <= 0:
                return float('inf')  # Infinite cross-entropy
            ce -= p_i * safe_log(q_i, base)
    
    return ce


def kl_divergence(p: List[float], q: List[float], base: float = 2.0) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(p || q).
    
    D_KL(p || q) = ∑ p(x) * log_base(p(x) / q(x))
    
    Args:
        p: True probability distribution
        q: Approximate probability distribution
        base: Logarithm base (2 for bits, e for nats)
        
    Returns:
        KL divergence from q to p (non-negative, asymmetric)
        
    Raises:
        ValueError: If distributions are invalid or mismatched lengths
        
    Examples:
        >>> kl_divergence([0.5, 0.5], [0.5, 0.5])  # Identical distributions
        0.0
        >>> kl_divergence([0.8, 0.2], [0.6, 0.4])  # Different distributions
        0.097
        >>> kl_divergence([1.0, 0.0], [0.5, 0.5])  # High divergence
        1.0
    """
    validate_distribution(p)
    validate_distribution(q)
    
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")
    
    if base <= 0 or base == 1:
        raise ValueError("Base must be positive and not equal to 1")
    
    kl = 0.0
    for p_i, q_i in zip(p, q):
        if p_i > 0:
            if q_i <= 0:
                return float('inf')  # Infinite divergence
            kl += p_i * safe_log(p_i / q_i, base)
    
    return kl


def js_divergence(p: List[float], q: List[float], base: float = 2.0) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric KL divergence).
    
    JS(p, q) = 0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)
    where m = 0.5 * (p + q)
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        base: Logarithm base (2 for bits, e for nats)
        
    Returns:
        Jensen-Shannon divergence (symmetric, bounded [0, 1] for base=2)
        
    Raises:
        ValueError: If distributions are invalid or mismatched lengths
        
    Examples:
        >>> js_divergence([0.5, 0.5], [0.5, 0.5])  # Identical
        0.0
        >>> js_divergence([1.0, 0.0], [0.0, 1.0])  # Opposite
        1.0
        >>> js_divergence([0.8, 0.2], [0.6, 0.4])  # Moderate difference
        0.024
    """
    validate_distribution(p)
    validate_distribution(q)
    
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")
    
    # Compute mixture distribution m = 0.5 * (p + q)
    m = [(p_i + q_i) / 2.0 for p_i, q_i in zip(p, q)]
    
    # JS divergence is average of KL divergences to mixture
    js = 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)
    
    return js


def renyi_entropy(p: List[float], alpha: float, base: float = 2.0) -> float:
    """
    Compute Rényi entropy of order α.
    
    H_α(X) = (1 / (1 - α)) * log_base(∑ p(x)^α)
    
    Special cases:
    - α → 0: log(|support|) (Hartley entropy)
    - α → 1: Shannon entropy (limit)
    - α = 2: Collision entropy
    - α → ∞: min-entropy
    
    Args:
        p: Probability distribution
        alpha: Order parameter (α ≥ 0, α ≠ 1)
        base: Logarithm base
        
    Returns:
        Rényi entropy of order α
        
    Raises:
        ValueError: If α = 1 or distributions invalid
        
    Examples:
        >>> renyi_entropy([0.5, 0.5], 0)  # Hartley entropy
        1.0
        >>> renyi_entropy([0.5, 0.5], 2)  # Collision entropy
        1.0
        >>> renyi_entropy([0.8, 0.2], 2)  # Skewed distribution
        0.678
    """
    validate_distribution(p)
    
    if alpha < 0:
        raise ValueError("Alpha must be non-negative")
    if alpha == 1:
        raise ValueError("Use entropy() function for α = 1 (Shannon entropy)")
    if base <= 0 or base == 1:
        raise ValueError("Base must be positive and not equal to 1")
    
    if alpha == 0:
        # Hartley entropy: log of support size
        support_size = sum(1 for prob in p if prob > 0)
        return safe_log(support_size, base)
    
    if math.isinf(alpha):
        # Min-entropy: -log(max(p))
        max_prob = max(p)
        return -safe_log(max_prob, base)
    
    # General case: H_α(X) = (1 / (1 - α)) * log(∑ p^α)
    sum_p_alpha = sum(prob ** alpha for prob in p if prob > 0)
    
    if sum_p_alpha <= 0:
        return float('inf')
    
    return safe_log(sum_p_alpha, base) / (1 - alpha)


def tsallis_entropy(p: List[float], q: float, base: float = 2.0) -> float:
    """
    Compute Tsallis entropy (non-extensive generalization).
    
    S_q(X) = (1 / (q - 1)) * (1 - ∑ p(x)^q)
    
    Special cases:
    - q → 1: Shannon entropy (limit)
    - q = 2: Gini-Simpson index related
    
    Args:
        p: Probability distribution
        q: Entropic index (q > 0, q ≠ 1)
        base: Logarithm base (affects normalization)
        
    Returns:
        Tsallis entropy of index q
        
    Raises:
        ValueError: If q = 1 or distributions invalid
        
    Examples:
        >>> tsallis_entropy([0.5, 0.5], 2)  # q = 2
        0.5
        >>> tsallis_entropy([0.25, 0.25, 0.25, 0.25], 2)  # Uniform
        0.75
        >>> tsallis_entropy([1.0, 0.0], 2)  # Deterministic
        0.0
    """
    validate_distribution(p)
    
    if q <= 0:
        raise ValueError("q must be positive")
    if q == 1:
        raise ValueError("Use entropy() function for q = 1 (Shannon entropy)")
    
    # Compute ∑ p^q
    sum_p_q = sum(prob ** q for prob in p if prob > 0)
    
    # Tsallis entropy: (1 - ∑ p^q) / (q - 1)
    tsallis = (1 - sum_p_q) / (q - 1)
    
    return tsallis
