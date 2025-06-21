# MetricaX Bayesian Statistics Module 🎯

**Production-ready Bayesian inference toolkit for data scientists and researchers**

This module provides comprehensive tools for Bayesian statistical analysis, featuring mathematically rigorous implementations with real-world applications.

## 🚀 **Overview**

The Bayesian module implements fundamental concepts in Bayesian statistics with a focus on:
- **Conjugate Prior Systems** - Analytical posterior updates
- **Beta Distribution Family** - Complete statistical analysis
- **Bayes' Theorem Applications** - Discrete and continuous inference
- **Numerical Stability** - Robust implementations for production use

## 📊 **Function Categories**

### **Beta Distributions (5 functions)**
Complete statistical analysis of Beta distributions for proportion modeling:

| Function | Description | Use Case |
|----------|-------------|----------|
| `beta_pdf(x, alpha, beta)` | Probability density function | Likelihood evaluation |
| `beta_cdf(x, alpha, beta)` | Cumulative distribution | Probability calculations |
| `beta_mean(alpha, beta)` | Distribution mean | Point estimates |
| `beta_var(alpha, beta)` | Distribution variance | Uncertainty quantification |
| `beta_mode(alpha, beta)` | Distribution mode | Most likely value |

### **Bayes' Theorem (4 functions)**
Core Bayesian inference for updating beliefs:

| Function | Description | Mathematical Form |
|----------|-------------|-------------------|
| `bayes_posterior(prior, likelihood, marginal)` | Basic Bayes rule | P(H\|E) = P(E\|H)P(H)/P(E) |
| `bayes_odds(prior_odds, likelihood_ratio)` | Odds form | Posterior Odds = Prior Odds × LR |
| `bayes_update_discrete(priors, likelihoods)` | Multi-hypothesis update | Normalized posteriors |
| `marginal_likelihood_discrete(priors, likelihoods)` | Evidence calculation | P(E) = ΣP(E\|Hᵢ)P(Hᵢ) |

### **Conjugate Priors (3 functions)**
Analytical posterior updates for common likelihood families:

| Function | Conjugate Pair | Application |
|----------|----------------|-------------|
| `update_beta_binomial(α, β, successes, failures)` | Beta-Binomial | Conversion rates, success rates |
| `update_normal_known_variance(μ₀, σ₀, data, σ)` | Normal-Normal | Sensor calibration, measurements |
| `update_poisson_gamma(α, β, observed_sum, n_obs)` | Gamma-Poisson | Event rates, count data |

### **Utilities (4 functions)**
Mathematical support functions with numerical stability:

| Function | Purpose | Features |
|----------|---------|----------|
| `gamma_func(x)` | Gamma function evaluation | Input validation, edge cases |
| `validate_prob(x)` | Probability validation | Range checking, finite values |
| `normalize(probs)` | Probability normalization | Sum-to-one constraint |
| `safe_div(a, b, default)` | Safe division | Zero-division handling |

## 🎯 **Real-World Applications**

### **A/B Testing Analysis**
```python
import metricax.bayesian as mb

# Update beliefs with conversion data
control_alpha, control_beta = mb.update_beta_binomial(1, 1, 12, 108)
treatment_alpha, treatment_beta = mb.update_beta_binomial(1, 1, 15, 85)

# Compare conversion rates
control_rate = mb.beta_mean(control_alpha, control_beta)
treatment_rate = mb.beta_mean(treatment_alpha, treatment_beta)

print(f"Control: {control_rate:.3f}, Treatment: {treatment_rate:.3f}")
```

### **Spam Classification**
```python
# Bayesian spam filter
priors = [0.4, 0.6]  # P(spam), P(ham)
likelihoods = [0.8, 0.1]  # P(word|spam), P(word|ham)
posteriors = mb.bayes_update_discrete(priors, likelihoods)

spam_probability = posteriors[0]
```

### **Quality Control**
```python
# Monitor defect rates with streaming data
alpha, beta = 2, 98  # Prior belief: ~2% defect rate

# Update with each batch
for items, defects in batches:
    alpha, beta = mb.update_beta_binomial(alpha, beta, defects, items - defects)
    current_rate = mb.beta_mean(alpha, beta)
    
    if current_rate > 0.03:  # Alert threshold
        print("⚠️ Quality alert!")
```

## 📈 **Mathematical Foundation**

### **Beta Distribution**
The Beta distribution Beta(α, β) is the conjugate prior for the Binomial likelihood:

- **PDF**: `f(x; α, β) = (Γ(α+β)/(Γ(α)Γ(β))) × x^(α-1) × (1-x)^(β-1)`
- **Mean**: `α/(α+β)`
- **Variance**: `αβ/((α+β)²(α+β+1))`
- **Mode**: `(α-1)/(α+β-2)` for α,β > 1

### **Conjugate Prior Updates**
For conjugate prior-likelihood pairs, posterior parameters update analytically:

1. **Beta-Binomial**: `Beta(α, β) + Binomial(n, k) → Beta(α+k, β+n-k)`
2. **Normal-Normal**: `N(μ₀, σ₀²) + N(μ, σ²) → N(μₙ, σₙ²)` (known variance)
3. **Gamma-Poisson**: `Gamma(α, β) + Poisson(λ) → Gamma(α+Σx, β+n)`

## 🧪 **Examples and Testing**

### **Interactive Examples**
```python
# Run comprehensive examples
from metricax.bayesian.examples import ab_testing, spam_filter, data_updates

ab_testing.run_example()      # A/B testing analysis
spam_filter.run_example()    # Bayesian spam filter
data_updates.run_example()   # Online learning scenarios
```

### **Unit Testing**
```bash
# Run the test suite
python -m pytest metricax/bayesian/tests/ -v
```

## 🔬 **Advanced Features**

### **Numerical Stability**
- Log-space computations for extreme parameter values
- Epsilon handling for zero probabilities
- Overflow protection in Beta function calculations

### **Input Validation**
- Comprehensive parameter checking
- Informative error messages
- Edge case handling

### **Performance Optimizations**
- Efficient algorithms for common cases
- Minimal memory footprint
- Pure Python implementation (no heavy dependencies)

## 📚 **References**

- **Gelman, A. et al.** (2013). *Bayesian Data Analysis*. 3rd Edition.
- **Bishop, C.M.** (2006). *Pattern Recognition and Machine Learning*.
- **MacKay, D.J.C.** (2003). *Information Theory, Inference and Learning Algorithms*.

## 🤝 **Contributing**

See the main [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines on adding new Bayesian functions or improving existing implementations.

---

**Part of MetricaX - Professional Mathematical Toolkit for Python**
