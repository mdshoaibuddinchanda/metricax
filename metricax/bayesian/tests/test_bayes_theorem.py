"""
Unit tests for metricax.bayesian.bayes_theorem module.
"""

import pytest
import math
import sys
import os

# Add the parent directory to the path to import metricax
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import metricax.bayesian as mb


class TestBayesPosterior:
    """Test cases for bayes_posterior"""
    
    def test_bayes_posterior_basic(self):
        """Test basic Bayes' theorem calculation"""
        # P(H|E) = P(E|H) * P(H) / P(E)
        prior = 0.3
        likelihood = 0.8
        marginal = 0.5
        
        expected = (likelihood * prior) / marginal
        result = mb.bayes_posterior(prior, likelihood, marginal)
        
        assert abs(result - expected) < 1e-10
    
    def test_bayes_posterior_extreme_cases(self):
        """Test extreme probability cases"""
        # Certain evidence
        assert mb.bayes_posterior(0.5, 1.0, 1.0) == 0.5
        
        # Impossible evidence given hypothesis
        assert mb.bayes_posterior(0.5, 0.0, 0.5) == 0.0
        
        # Certain hypothesis
        assert mb.bayes_posterior(1.0, 0.8, 0.8) == 1.0
    
    def test_bayes_posterior_invalid_input(self):
        """Test invalid inputs"""
        with pytest.raises(ValueError):
            mb.bayes_posterior(-0.1, 0.8, 0.5)  # Invalid prior
        
        with pytest.raises(ValueError):
            mb.bayes_posterior(0.3, 1.1, 0.5)   # Invalid likelihood
        
        with pytest.raises(ValueError):
            mb.bayes_posterior(0.3, 0.8, 0.0)   # Zero marginal


class TestBayesOdds:
    """Test cases for bayes_odds"""
    
    def test_bayes_odds_basic(self):
        """Test basic odds calculation"""
        prior_odds = 1.0  # 50-50 prior
        likelihood_ratio = 2.0
        
        expected = prior_odds * likelihood_ratio
        result = mb.bayes_odds(prior_odds, likelihood_ratio)
        
        assert abs(result - expected) < 1e-10
    
    def test_bayes_odds_extreme_cases(self):
        """Test extreme cases"""
        # No evidence (likelihood ratio = 1)
        assert mb.bayes_odds(2.0, 1.0) == 2.0
        
        # Strong evidence
        assert mb.bayes_odds(1.0, 10.0) == 10.0
        
        # Zero prior odds
        assert mb.bayes_odds(0.0, 5.0) == 0.0
    
    def test_bayes_odds_invalid_input(self):
        """Test invalid inputs"""
        with pytest.raises(ValueError):
            mb.bayes_odds(-1.0, 2.0)  # Negative prior odds
        
        with pytest.raises(ValueError):
            mb.bayes_odds(1.0, -2.0)  # Negative likelihood ratio
        
        with pytest.raises(ValueError):
            mb.bayes_odds(float('inf'), 2.0)  # Infinite input


class TestBayesUpdateDiscrete:
    """Test cases for bayes_update_discrete"""
    
    def test_bayes_update_discrete_basic(self):
        """Test basic discrete update"""
        priors = [0.3, 0.7]
        likelihoods = [0.8, 0.2]
        
        result = mb.bayes_update_discrete(priors, likelihoods)
        
        # Check that result sums to 1
        assert abs(sum(result) - 1.0) < 1e-10
        
        # Check that all probabilities are valid
        assert all(0 <= p <= 1 for p in result)
        
        # Manual calculation
        marginal = 0.3 * 0.8 + 0.7 * 0.2  # 0.38
        expected = [(0.8 * 0.3) / marginal, (0.2 * 0.7) / marginal]
        
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-10
    
    def test_bayes_update_discrete_three_hypotheses(self):
        """Test with three hypotheses"""
        priors = [0.25, 0.25, 0.5]
        likelihoods = [0.9, 0.1, 0.3]
        
        result = mb.bayes_update_discrete(priors, likelihoods)
        
        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-10
        assert all(0 <= p <= 1 for p in result)
    
    def test_bayes_update_discrete_invalid_input(self):
        """Test invalid inputs"""
        # Mismatched lengths
        with pytest.raises(ValueError):
            mb.bayes_update_discrete([0.5, 0.5], [0.8])
        
        # Priors don't sum to 1
        with pytest.raises(ValueError):
            mb.bayes_update_discrete([0.3, 0.3], [0.8, 0.2])
        
        # Invalid probabilities
        with pytest.raises(ValueError):
            mb.bayes_update_discrete([0.5, 0.5], [1.1, 0.2])


class TestMarginalLikelihoodDiscrete:
    """Test cases for marginal_likelihood_discrete"""
    
    def test_marginal_likelihood_basic(self):
        """Test basic marginal likelihood calculation"""
        priors = [0.3, 0.7]
        likelihoods = [0.8, 0.2]
        
        expected = 0.3 * 0.8 + 0.7 * 0.2  # 0.38
        result = mb.marginal_likelihood_discrete(priors, likelihoods)
        
        assert abs(result - expected) < 1e-10
    
    def test_marginal_likelihood_multiple_hypotheses(self):
        """Test with multiple hypotheses"""
        priors = [0.2, 0.3, 0.5]
        likelihoods = [0.9, 0.5, 0.1]
        
        expected = 0.2 * 0.9 + 0.3 * 0.5 + 0.5 * 0.1
        result = mb.marginal_likelihood_discrete(priors, likelihoods)
        
        assert abs(result - expected) < 1e-10
    
    def test_marginal_likelihood_boundary_cases(self):
        """Test boundary cases"""
        # All likelihood on one hypothesis
        priors = [0.5, 0.5]
        likelihoods = [1.0, 0.0]
        
        result = mb.marginal_likelihood_discrete(priors, likelihoods)
        assert abs(result - 0.5) < 1e-10
    
    def test_marginal_likelihood_invalid_input(self):
        """Test invalid inputs"""
        with pytest.raises(ValueError):
            mb.marginal_likelihood_discrete([], [])
        
        with pytest.raises(ValueError):
            mb.marginal_likelihood_discrete([0.5], [0.8, 0.2])


if __name__ == "__main__":
    pytest.main([__file__])
