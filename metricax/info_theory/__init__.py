"""
MetricaX Information Theory Module

This module provides comprehensive tools for information-theoretic analysis,
including entropy measures, mutual information, coding theory, and distribution distances.
"""

# Entropy and Variants (6 functions)
from .entropy import (
    entropy,
    cross_entropy,
    kl_divergence,
    js_divergence,
    renyi_entropy,
    tsallis_entropy,
)

# Mutual Information and Dependence Measures (7 functions)
from .mutual_info import (
    mutual_information,
    conditional_entropy,
    information_gain,
    symmetric_uncertainty,
    variation_of_information,
    total_correlation,
    multi_information,
)

# Code Length & Optimal Coding (3 functions)
from .coding_theory import (
    optimal_code_length,
    fano_inequality,
    redundancy,
)

# Distance Measures Between Distributions (4 functions)
from .distance_measures import (
    hellinger_distance,
    total_variation_distance,
    bhattacharyya_distance,
    wasserstein_distance_1d,
)

# Utility Functions (4 functions)
from .utils import (
    validate_distribution,
    normalize_distribution,
    joint_distribution,
    safe_log,
)

__all__ = [
    # Entropy and variants
    "entropy",
    "cross_entropy",
    "kl_divergence",
    "js_divergence",
    "renyi_entropy",
    "tsallis_entropy",
    
    # Mutual information and dependence
    "mutual_information",
    "conditional_entropy",
    "information_gain",
    "symmetric_uncertainty",
    "variation_of_information",
    "total_correlation",
    "multi_information",
    
    # Coding theory
    "optimal_code_length",
    "fano_inequality",
    "redundancy",
    
    # Distance measures
    "hellinger_distance",
    "total_variation_distance",
    "bhattacharyya_distance",
    "wasserstein_distance_1d",
    
    # Utilities
    "validate_distribution",
    "normalize_distribution",
    "joint_distribution",
    "safe_log",
]
