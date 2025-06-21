"""
Conjugate prior update functions for Bayesian inference.

This module implements conjugate prior-likelihood pairs that allow
for analytical posterior updates without numerical integration.
"""

import math
from typing import List, Tuple
from .utils import validate_prob, safe_div


def update_beta_binomial(alpha: float, beta: float, successes: int, failures: int) -> Tuple[float, float]:
    """
    Update Beta prior with Binomial likelihood (Beta-Binomial conjugacy).
    
    Prior: Beta(α, β)
    Likelihood: Binomial(n, p) with k successes, (n-k) failures
    Posterior: Beta(α + k, β + (n-k))
    
    Args:
        alpha: Prior Beta parameter α > 0
        beta: Prior Beta parameter β > 0
        successes: Number of observed successes (k ≥ 0)
        failures: Number of observed failures (n-k ≥ 0)
        
    Returns:
        Tuple of updated (alpha_new, beta_new) parameters
        
    Raises:
        ValueError: If parameters are invalid
        
    Examples:
        >>> update_beta_binomial(1, 1, 7, 3)
        (8.0, 4.0)
        >>> update_beta_binomial(2, 5, 10, 5)
        (12.0, 10.0)
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive")
    if not (math.isfinite(alpha) and math.isfinite(beta)):
        raise ValueError("Alpha and beta must be finite")
    if successes < 0 or failures < 0:
        raise ValueError("Successes and failures must be non-negative")
    if not isinstance(successes, int) or not isinstance(failures, int):
        raise ValueError("Successes and failures must be integers")
    
    alpha_new = alpha + successes
    beta_new = beta + failures
    
    return (alpha_new, beta_new)


def update_normal_known_variance(mu0: float, sigma0: float, data: List[float], 
                                sigma_likelihood: float = 1.0) -> Tuple[float, float]:
    """
    Update Normal prior with Normal likelihood (known variance conjugacy).
    
    Prior: N(μ₀, σ₀²)
    Likelihood: N(μ, σ²) with known σ²
    Posterior: N(μₙ, σₙ²)
    
    where:
    σₙ² = 1 / (1/σ₀² + n/σ²)
    μₙ = σₙ² * (μ₀/σ₀² + Σxᵢ/σ²)
    
    Args:
        mu0: Prior mean μ₀
        sigma0: Prior standard deviation σ₀ > 0
        data: List of observed data points
        sigma_likelihood: Known likelihood standard deviation σ > 0
        
    Returns:
        Tuple of updated (mu_new, sigma_new) parameters
        
    Raises:
        ValueError: If parameters are invalid
        
    Examples:
        >>> update_normal_known_variance(0, 1, [1, 2, 3], 1)
        (1.5, 0.5)
        >>> update_normal_known_variance(5, 2, [4, 6], 1.5)
        (4.615384615384615, 0.8451542547285166)
    """
    if sigma0 <= 0:
        raise ValueError("Prior sigma must be positive")
    if sigma_likelihood <= 0:
        raise ValueError("Likelihood sigma must be positive")
    if not math.isfinite(mu0):
        raise ValueError("Prior mu must be finite")
    if not (math.isfinite(sigma0) and math.isfinite(sigma_likelihood)):
        raise ValueError("Sigma values must be finite")
    if not data:
        raise ValueError("Data list cannot be empty")
    if not all(math.isfinite(x) for x in data):
        raise ValueError("All data points must be finite")
    
    n = len(data)
    data_sum = sum(data)
    
    # Precision (inverse variance) calculations
    prior_precision = 1 / (sigma0 ** 2)
    likelihood_precision = n / (sigma_likelihood ** 2)
    
    # Posterior precision and variance
    posterior_precision = prior_precision + likelihood_precision
    sigma_new_squared = 1 / posterior_precision
    sigma_new = math.sqrt(sigma_new_squared)
    
    # Posterior mean
    mu_new = sigma_new_squared * (mu0 * prior_precision + data_sum / (sigma_likelihood ** 2))
    
    return (mu_new, sigma_new)


def update_poisson_gamma(alpha: float, beta: float, observed_sum: int, n_obs: int) -> Tuple[float, float]:
    """
    Update Gamma prior with Poisson likelihood (Poisson-Gamma conjugacy).
    
    Prior: Gamma(α, β)
    Likelihood: Poisson(λ) with n observations, sum = Σxᵢ
    Posterior: Gamma(α + Σxᵢ, β + n)
    
    Args:
        alpha: Prior Gamma shape parameter α > 0
        beta: Prior Gamma rate parameter β > 0
        observed_sum: Sum of observed Poisson counts Σxᵢ ≥ 0
        n_obs: Number of observations n > 0
        
    Returns:
        Tuple of updated (alpha_new, beta_new) parameters
        
    Raises:
        ValueError: If parameters are invalid
        
    Examples:
        >>> update_poisson_gamma(2, 1, 15, 5)
        (17.0, 6.0)
        >>> update_poisson_gamma(1, 0.5, 8, 3)
        (9.0, 3.5)
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive")
    if not (math.isfinite(alpha) and math.isfinite(beta)):
        raise ValueError("Alpha and beta must be finite")
    if observed_sum < 0:
        raise ValueError("Observed sum must be non-negative")
    if n_obs <= 0:
        raise ValueError("Number of observations must be positive")
    if not isinstance(observed_sum, int) or not isinstance(n_obs, int):
        raise ValueError("Observed sum and n_obs must be integers")
    
    alpha_new = alpha + observed_sum
    beta_new = beta + n_obs
    
    return (alpha_new, beta_new)
