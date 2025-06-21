"""
MetricaX Bayesian Statistics Module

This module provides comprehensive tools for Bayesian statistical analysis,
including Beta distributions, Bayes' theorem applications, and conjugate priors.
"""

from .beta_distributions import (
    beta_pdf,
    beta_cdf,
    beta_mean,
    beta_var,
    beta_mode,
)

from .bayes_theorem import (
    bayes_posterior,
    bayes_odds,
    bayes_update_discrete,
    marginal_likelihood_discrete,
)

from .conjugate_priors import (
    update_beta_binomial,
    update_normal_known_variance,
    update_poisson_gamma,
)

from .utils import (
    gamma_func,
    validate_prob,
    normalize,
    safe_div,
)

__all__ = [
    # Beta distribution functions
    "beta_pdf",
    "beta_cdf",
    "beta_mean",
    "beta_var",
    "beta_mode",

    # Bayes theorem functions
    "bayes_posterior",
    "bayes_odds",
    "bayes_update_discrete",
    "marginal_likelihood_discrete",

    # Conjugate prior functions
    "update_beta_binomial",
    "update_normal_known_variance",
    "update_poisson_gamma",

    # Utility functions
    "gamma_func",
    "validate_prob",
    "normalize",
    "safe_div",
]