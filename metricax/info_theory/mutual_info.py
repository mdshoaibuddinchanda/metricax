"""
Mutual Information and Dependence Measures.

This module implements mutual information, conditional entropy,
and related measures for quantifying statistical dependence.
"""

import math
from typing import List, Union
from .utils import validate_distribution, safe_log
from .entropy import entropy


def mutual_information(p_xy: List[List[float]], p_x: List[float], p_y: List[float], 
                      base: float = 2.0) -> float:
    """
    Compute mutual information I(X; Y).
    
    I(X; Y) = ∑∑ p(x,y) * log(p(x,y) / (p(x) * p(y)))
    
    Args:
        p_xy: Joint distribution P(X, Y) as 2D list
        p_x: Marginal distribution P(X)
        p_y: Marginal distribution P(Y)
        base: Logarithm base (2 for bits, e for nats)
        
    Returns:
        Mutual information I(X; Y) ≥ 0
        
    Raises:
        ValueError: If distributions are invalid or incompatible
        
    Examples:
        >>> # Independent variables
        >>> p_xy = [[0.25, 0.25], [0.25, 0.25]]
        >>> p_x = [0.5, 0.5]
        >>> p_y = [0.5, 0.5]
        >>> mutual_information(p_xy, p_x, p_y)
        0.0
        
        >>> # Perfectly dependent variables
        >>> p_xy = [[0.5, 0.0], [0.0, 0.5]]
        >>> mutual_information(p_xy, p_x, p_y)
        1.0
    """
    validate_distribution(p_x)
    validate_distribution(p_y)
    
    # Validate joint distribution
    if len(p_xy) != len(p_x):
        raise ValueError("Joint distribution rows must match p_x size")
    if not all(len(row) == len(p_y) for row in p_xy):
        raise ValueError("Joint distribution columns must match p_y size")
    
    # Flatten and validate joint distribution
    p_xy_flat = [prob for row in p_xy for prob in row]
    validate_distribution(p_xy_flat)
    
    mi = 0.0
    for i, p_x_i in enumerate(p_x):
        for j, p_y_j in enumerate(p_y):
            p_xy_ij = p_xy[i][j]
            if p_xy_ij > 0 and p_x_i > 0 and p_y_j > 0:
                mi += p_xy_ij * safe_log(p_xy_ij / (p_x_i * p_y_j), base)
    
    return max(0.0, mi)  # Ensure non-negative due to numerical errors


def conditional_entropy(p_xy: List[List[float]], p_y: List[float], 
                       base: float = 2.0) -> float:
    """
    Compute conditional entropy H(X|Y).
    
    H(X|Y) = -∑∑ p(x,y) * log(p(x|y))
           = -∑∑ p(x,y) * log(p(x,y) / p(y))
    
    Args:
        p_xy: Joint distribution P(X, Y) as 2D list
        p_y: Marginal distribution P(Y)
        base: Logarithm base
        
    Returns:
        Conditional entropy H(X|Y) ≥ 0
        
    Examples:
        >>> # X completely determined by Y
        >>> p_xy = [[0.5, 0.0], [0.0, 0.5]]
        >>> p_y = [0.5, 0.5]
        >>> conditional_entropy(p_xy, p_y)
        0.0
        
        >>> # X independent of Y
        >>> p_xy = [[0.25, 0.25], [0.25, 0.25]]
        >>> conditional_entropy(p_xy, p_y)
        1.0
    """
    validate_distribution(p_y)
    
    if not all(len(row) == len(p_y) for row in p_xy):
        raise ValueError("Joint distribution columns must match p_y size")
    
    # Flatten and validate joint distribution
    p_xy_flat = [prob for row in p_xy for prob in row]
    validate_distribution(p_xy_flat)
    
    h_x_given_y = 0.0
    for i, row in enumerate(p_xy):
        for j, (p_xy_ij, p_y_j) in enumerate(zip(row, p_y)):
            if p_xy_ij > 0 and p_y_j > 0:
                # p(x|y) = p(x,y) / p(y)
                p_x_given_y = p_xy_ij / p_y_j
                h_x_given_y -= p_xy_ij * safe_log(p_x_given_y, base)
    
    return max(0.0, h_x_given_y)


def information_gain(p_prior: List[float], p_posterior: List[float], 
                    base: float = 2.0) -> float:
    """
    Compute information gain (reduction in entropy).
    
    IG = H(prior) - H(posterior)
    
    Args:
        p_prior: Prior probability distribution
        p_posterior: Posterior probability distribution after observation
        base: Logarithm base
        
    Returns:
        Information gain ≥ 0
        
    Examples:
        >>> # Complete information gain
        >>> p_prior = [0.5, 0.5]
        >>> p_posterior = [1.0, 0.0]
        >>> information_gain(p_prior, p_posterior)
        1.0
        
        >>> # No information gain
        >>> information_gain([0.5, 0.5], [0.5, 0.5])
        0.0
    """
    h_prior = entropy(p_prior, base)
    h_posterior = entropy(p_posterior, base)
    
    return max(0.0, h_prior - h_posterior)


def symmetric_uncertainty(p_xy: List[List[float]], p_x: List[float], 
                         p_y: List[float], base: float = 2.0) -> float:
    """
    Compute symmetric uncertainty (normalized mutual information).
    
    SU(X, Y) = 2 * I(X; Y) / (H(X) + H(Y))
    
    Args:
        p_xy: Joint distribution P(X, Y)
        p_x: Marginal distribution P(X)
        p_y: Marginal distribution P(Y)
        base: Logarithm base
        
    Returns:
        Symmetric uncertainty ∈ [0, 1]
        
    Examples:
        >>> # Independent variables
        >>> p_xy = [[0.25, 0.25], [0.25, 0.25]]
        >>> p_x = [0.5, 0.5]
        >>> p_y = [0.5, 0.5]
        >>> symmetric_uncertainty(p_xy, p_x, p_y)
        0.0
        
        >>> # Perfectly dependent
        >>> p_xy = [[0.5, 0.0], [0.0, 0.5]]
        >>> symmetric_uncertainty(p_xy, p_x, p_y)
        1.0
    """
    mi = mutual_information(p_xy, p_x, p_y, base)
    h_x = entropy(p_x, base)
    h_y = entropy(p_y, base)
    
    if h_x + h_y == 0:
        return 0.0  # Both variables are deterministic
    
    return 2 * mi / (h_x + h_y)


def variation_of_information(p_xy: List[List[float]], p_x: List[float], 
                           p_y: List[float], base: float = 2.0) -> float:
    """
    Compute variation of information (VI distance metric).
    
    VI(X, Y) = H(X|Y) + H(Y|X) = H(X) + H(Y) - 2*I(X; Y)
    
    Args:
        p_xy: Joint distribution P(X, Y)
        p_x: Marginal distribution P(X)
        p_y: Marginal distribution P(Y)
        base: Logarithm base
        
    Returns:
        Variation of information ≥ 0 (metric distance)
        
    Examples:
        >>> # Independent variables
        >>> p_xy = [[0.25, 0.25], [0.25, 0.25]]
        >>> p_x = [0.5, 0.5]
        >>> p_y = [0.5, 0.5]
        >>> variation_of_information(p_xy, p_x, p_y)
        2.0
        
        >>> # Identical variables
        >>> p_xy = [[0.5, 0.0], [0.0, 0.5]]
        >>> variation_of_information(p_xy, p_x, p_y)
        0.0
    """
    h_x = entropy(p_x, base)
    h_y = entropy(p_y, base)
    mi = mutual_information(p_xy, p_x, p_y, base)
    
    return h_x + h_y - 2 * mi


def total_correlation(p_xyz: List[List[List[float]]], 
                     p_x: List[float], p_y: List[float], p_z: List[float],
                     base: float = 2.0) -> float:
    """
    Compute total correlation (multivariate generalization of mutual information).
    
    TC(X, Y, Z) = H(X) + H(Y) + H(Z) - H(X, Y, Z)
    
    Args:
        p_xyz: Joint distribution P(X, Y, Z) as 3D list
        p_x: Marginal distribution P(X)
        p_y: Marginal distribution P(Y)
        p_z: Marginal distribution P(Z)
        base: Logarithm base
        
    Returns:
        Total correlation ≥ 0
        
    Examples:
        >>> # Independent variables
        >>> p_xyz = [[[0.125, 0.125], [0.125, 0.125]], 
        ...           [[0.125, 0.125], [0.125, 0.125]]]
        >>> p_x = [0.5, 0.5]
        >>> p_y = [0.5, 0.5]
        >>> p_z = [0.5, 0.5]
        >>> total_correlation(p_xyz, p_x, p_y, p_z)
        0.0
    """
    validate_distribution(p_x)
    validate_distribution(p_y)
    validate_distribution(p_z)
    
    # Validate 3D joint distribution
    if len(p_xyz) != len(p_x):
        raise ValueError("First dimension must match p_x size")
    if not all(len(plane) == len(p_y) for plane in p_xyz):
        raise ValueError("Second dimension must match p_y size")
    if not all(len(row) == len(p_z) for plane in p_xyz for row in plane):
        raise ValueError("Third dimension must match p_z size")
    
    # Flatten and validate joint distribution
    p_xyz_flat = [prob for plane in p_xyz for row in plane for prob in row]
    validate_distribution(p_xyz_flat)
    
    # Compute marginal entropies
    h_x = entropy(p_x, base)
    h_y = entropy(p_y, base)
    h_z = entropy(p_z, base)
    
    # Compute joint entropy H(X, Y, Z)
    h_xyz = entropy(p_xyz_flat, base)
    
    return h_x + h_y + h_z - h_xyz


def multi_information(p_multivariate: List[float], marginals: List[List[float]], 
                     base: float = 2.0) -> float:
    """
    Compute multi-information (alternative form of total correlation).
    
    I(X₁; X₂; ...; Xₙ) = ∑ H(Xᵢ) - H(X₁, X₂, ..., Xₙ)
    
    Args:
        p_multivariate: Flattened joint distribution
        marginals: List of marginal distributions
        base: Logarithm base
        
    Returns:
        Multi-information ≥ 0
        
    Examples:
        >>> # Two independent binary variables
        >>> p_joint = [0.25, 0.25, 0.25, 0.25]
        >>> marginals = [[0.5, 0.5], [0.5, 0.5]]
        >>> multi_information(p_joint, marginals)
        0.0
    """
    validate_distribution(p_multivariate)
    
    for i, marginal in enumerate(marginals):
        validate_distribution(marginal)
    
    # Sum of marginal entropies
    sum_marginal_entropies = sum(entropy(marginal, base) for marginal in marginals)
    
    # Joint entropy
    joint_entropy = entropy(p_multivariate, base)
    
    return sum_marginal_entropies - joint_entropy
