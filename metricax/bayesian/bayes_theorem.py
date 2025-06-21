"""
Core Bayes theorem implementations for Bayesian inference.

This module provides functions for applying Bayes' rule in various forms,
including discrete and continuous cases.
"""

import math
from typing import List, Tuple
from .utils import validate_prob, normalize, safe_div


def bayes_posterior(prior: float, likelihood: float, marginal: float) -> float:
    """
    Compute posterior probability using Bayes' theorem.
    
    P(H|E) = P(E|H) * P(H) / P(E)
    
    Args:
        prior: Prior probability P(H)
        likelihood: Likelihood P(E|H)
        marginal: Marginal probability P(E)
        
    Returns:
        Posterior probability P(H|E)
        
    Raises:
        ValueError: If probabilities are invalid
        
    Examples:
        >>> bayes_posterior(0.3, 0.8, 0.5)
        0.48
        >>> bayes_posterior(0.1, 0.9, 0.2)
        0.45
    """
    if not validate_prob(prior):
        raise ValueError("Prior must be a valid probability")
    if not validate_prob(likelihood):
        raise ValueError("Likelihood must be a valid probability")
    if not validate_prob(marginal):
        raise ValueError("Marginal must be a valid probability")
    
    if marginal == 0:
        raise ValueError("Marginal probability cannot be zero")
    
    return (likelihood * prior) / marginal


def bayes_odds(prior_odds: float, likelihood_ratio: float) -> float:
    """
    Compute posterior odds using Bayes' theorem in odds form.
    
    Posterior Odds = Prior Odds × Likelihood Ratio
    
    Args:
        prior_odds: Prior odds P(H) / P(¬H)
        likelihood_ratio: Likelihood ratio P(E|H) / P(E|¬H)
        
    Returns:
        Posterior odds P(H|E) / P(¬H|E)
        
    Raises:
        ValueError: If inputs are invalid
        
    Examples:
        >>> bayes_odds(1.0, 2.0)
        2.0
        >>> bayes_odds(0.5, 3.0)
        1.5
    """
    if not math.isfinite(prior_odds) or prior_odds < 0:
        raise ValueError("Prior odds must be finite and non-negative")
    if not math.isfinite(likelihood_ratio) or likelihood_ratio < 0:
        raise ValueError("Likelihood ratio must be finite and non-negative")
    
    return prior_odds * likelihood_ratio


def bayes_update_discrete(priors: List[float], likelihoods: List[float]) -> List[float]:
    """
    Update discrete prior probabilities using Bayes' theorem.
    
    For each hypothesis i:
    P(H_i|E) = P(E|H_i) * P(H_i) / P(E)
    
    where P(E) = Σ P(E|H_j) * P(H_j)
    
    Args:
        priors: List of prior probabilities for each hypothesis
        likelihoods: List of likelihoods P(E|H_i) for each hypothesis
        
    Returns:
        List of posterior probabilities
        
    Raises:
        ValueError: If inputs are invalid or mismatched lengths
        
    Examples:
        >>> bayes_update_discrete([0.3, 0.7], [0.8, 0.2])
        [0.6315789473684211, 0.36842105263157887]
        >>> bayes_update_discrete([0.25, 0.25, 0.5], [0.9, 0.1, 0.3])
        [0.5294117647058824, 0.058823529411764705, 0.35294117647058826]
    """
    if len(priors) != len(likelihoods):
        raise ValueError("Priors and likelihoods must have same length")
    if not priors or not likelihoods:
        raise ValueError("Input lists cannot be empty")
    
    # Validate priors
    if not all(validate_prob(p) for p in priors):
        raise ValueError("All priors must be valid probabilities")
    if not math.isclose(sum(priors), 1.0, rel_tol=1e-9):
        raise ValueError("Priors must sum to 1")
    
    # Validate likelihoods
    if not all(validate_prob(l) for l in likelihoods):
        raise ValueError("All likelihoods must be valid probabilities")
    
    # Compute marginal likelihood P(E)
    marginal = marginal_likelihood_discrete(priors, likelihoods)
    
    if marginal == 0:
        raise ValueError("Marginal likelihood is zero - no valid update possible")
    
    # Compute posteriors
    posteriors = [(likelihood * prior) / marginal 
                  for prior, likelihood in zip(priors, likelihoods)]
    
    return posteriors


def marginal_likelihood_discrete(priors: List[float], likelihoods: List[float]) -> float:
    """
    Compute marginal likelihood for discrete hypotheses.
    
    P(E) = Σ P(E|H_i) * P(H_i)
    
    Args:
        priors: List of prior probabilities for each hypothesis
        likelihoods: List of likelihoods P(E|H_i) for each hypothesis
        
    Returns:
        Marginal likelihood P(E)
        
    Raises:
        ValueError: If inputs are invalid or mismatched lengths
        
    Examples:
        >>> marginal_likelihood_discrete([0.3, 0.7], [0.8, 0.2])
        0.38
        >>> marginal_likelihood_discrete([0.25, 0.25, 0.5], [0.9, 0.1, 0.3])
        0.425
    """
    if len(priors) != len(likelihoods):
        raise ValueError("Priors and likelihoods must have same length")
    if not priors or not likelihoods:
        raise ValueError("Input lists cannot be empty")
    
    # Validate priors
    if not all(validate_prob(p) for p in priors):
        raise ValueError("All priors must be valid probabilities")
    
    # Validate likelihoods
    if not all(validate_prob(l) for l in likelihoods):
        raise ValueError("All likelihoods must be valid probabilities")
    
    # Compute marginal likelihood
    marginal = sum(likelihood * prior 
                   for prior, likelihood in zip(priors, likelihoods))
    
    return marginal
