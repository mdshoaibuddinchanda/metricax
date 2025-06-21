"""
Beta distribution functions for Bayesian statistics.

This module implements probability density function, cumulative distribution function,
and statistical moments for the Beta distribution.
"""

import math
from typing import Union
from .utils import gamma_func, validate_prob, safe_div


def beta_pdf(x: float, alpha: float, beta: float) -> float:
    """
    Compute the probability density function of the Beta distribution.

    PDF(x; α, β) = (Γ(α + β) / (Γ(α) * Γ(β))) * x^(α-1) * (1-x)^(β-1)

    Args:
        x: Value at which to evaluate PDF (must be in [0, 1])
        alpha: Shape parameter α > 0
        beta: Shape parameter β > 0

    Returns:
        Probability density at x

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> beta_pdf(0.5, 2, 2)
        1.5
        >>> beta_pdf(0.3, 1, 1)
        1.0
    """
    if not validate_prob(x):
        raise ValueError("x must be in [0, 1]")
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive")
    if not (math.isfinite(alpha) and math.isfinite(beta)):
        raise ValueError("Alpha and beta must be finite")

    # Handle boundary cases
    if x == 0:
        return float('inf') if alpha < 1 else (1.0 if alpha == 1 else 0.0)
    if x == 1:
        return float('inf') if beta < 1 else (1.0 if beta == 1 else 0.0)

    # Compute beta function B(α, β) = Γ(α) * Γ(β) / Γ(α + β)
    try:
        log_beta_func = (math.lgamma(alpha) + math.lgamma(beta) -
                        math.lgamma(alpha + beta))

        # Compute log PDF for numerical stability
        log_pdf = ((alpha - 1) * math.log(x) +
                  (beta - 1) * math.log(1 - x) -
                  log_beta_func)

        return math.exp(log_pdf)
    except (ValueError, OverflowError):
        return 0.0


def beta_cdf(x: float, alpha: float, beta: float, steps: int = 1000) -> float:
    """
    Compute the cumulative distribution function of the Beta distribution.

    Uses numerical integration (trapezoidal rule) to approximate the CDF.

    Args:
        x: Upper limit of integration (must be in [0, 1])
        alpha: Shape parameter α > 0
        beta: Shape parameter β > 0
        steps: Number of integration steps (default: 1000)

    Returns:
        Cumulative probability P(X ≤ x)

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> beta_cdf(0.5, 2, 2)
        0.5
        >>> beta_cdf(0.0, 1, 1)
        0.0
        >>> beta_cdf(1.0, 1, 1)
        1.0
    """
    if not validate_prob(x):
        raise ValueError("x must be in [0, 1]")
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive")
    if steps <= 0:
        raise ValueError("Steps must be positive")

    # Boundary cases
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    # Numerical integration using trapezoidal rule
    dx = x / steps
    integral = 0.0

    # First point (t = 0)
    if alpha >= 1:
        integral += beta_pdf(0, alpha, beta) * 0.5

    # Interior points
    for i in range(1, steps):
        t = i * dx
        integral += beta_pdf(t, alpha, beta)

    # Last point (t = x)
    integral += beta_pdf(x, alpha, beta) * 0.5

    return integral * dx


def beta_mean(alpha: float, beta: float) -> float:
    """
    Compute the mean of the Beta distribution.

    Mean = α / (α + β)

    Args:
        alpha: Shape parameter α > 0
        beta: Shape parameter β > 0

    Returns:
        Mean of the distribution

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> beta_mean(2, 2)
        0.5
        >>> beta_mean(1, 3)
        0.25
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive")
    if not (math.isfinite(alpha) and math.isfinite(beta)):
        raise ValueError("Alpha and beta must be finite")

    return alpha / (alpha + beta)


def beta_var(alpha: float, beta: float) -> float:
    """
    Compute the variance of the Beta distribution.

    Variance = (α * β) / ((α + β)² * (α + β + 1))

    Args:
        alpha: Shape parameter α > 0
        beta: Shape parameter β > 0

    Returns:
        Variance of the distribution

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> beta_var(2, 2)
        0.05
        >>> beta_var(1, 1)
        0.08333333333333333
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive")
    if not (math.isfinite(alpha) and math.isfinite(beta)):
        raise ValueError("Alpha and beta must be finite")

    numerator = alpha * beta
    denominator = (alpha + beta) ** 2 * (alpha + beta + 1)

    return safe_div(numerator, denominator)


def beta_mode(alpha: float, beta: float) -> Union[float, None]:
    """
    Compute the mode of the Beta distribution.

    Mode = (α - 1) / (α + β - 2) if α > 1 and β > 1
    Mode is undefined if α ≤ 1 or β ≤ 1

    Args:
        alpha: Shape parameter α > 0
        beta: Shape parameter β > 0

    Returns:
        Mode of the distribution, or None if undefined

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> beta_mode(2, 2)
        0.5
        >>> beta_mode(3, 2)
        0.6666666666666666
        >>> beta_mode(1, 1)  # Returns None (mode undefined)
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive")
    if not (math.isfinite(alpha) and math.isfinite(beta)):
        raise ValueError("Alpha and beta must be finite")

    # Mode is undefined for α ≤ 1 or β ≤ 1
    if alpha <= 1 or beta <= 1:
        return None

    return (alpha - 1) / (alpha + beta - 2)