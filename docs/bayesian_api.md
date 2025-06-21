# Bayesian Statistics API Reference

Complete reference for MetricaX Bayesian Statistics module with mathematical formulas, parameters, and practical examples.

## Overview

The `metricax.bayesian` module provides 16 production-ready functions for Bayesian statistical analysis:

- **Beta Distributions** (5 functions): PDF, CDF, moments, and properties
- **Bayes' Theorem** (4 functions): Posterior inference and odds calculations  
- **Conjugate Priors** (3 functions): Analytical posterior updates
- **Utilities** (4 functions): Mathematical support and validation

## Beta Distribution Functions

### `beta_pdf(x, alpha, beta)`

Compute the probability density function of the Beta distribution.

**Formula**: `PDF(x; α, β) = (Γ(α + β) / (Γ(α) * Γ(β))) * x^(α-1) * (1-x)^(β-1)`

**Parameters**:
- `x: float` - Value at which to evaluate PDF (must be in [0, 1])
- `alpha: float` - Shape parameter α > 0
- `beta: float` - Shape parameter β > 0

**Returns**: `float` - Probability density at x

**Example**:
```python
import metricax.bayesian as mb

# Uniform distribution (α=1, β=1)
density = mb.beta_pdf(0.5, 1, 1)  # Returns: 1.0

# Bell-shaped distribution (α=2, β=2)  
density = mb.beta_pdf(0.5, 2, 2)  # Returns: 1.5
```

**Use Cases**:
- Modeling conversion rates, success probabilities
- Bayesian A/B testing prior/posterior distributions
- Quality control and reliability analysis

---

### `beta_cdf(x, alpha, beta, steps=1000)`

Compute the cumulative distribution function of the Beta distribution.

**Formula**: `CDF(x; α, β) = ∫₀ˣ PDF(t; α, β) dt`

**Parameters**:
- `x: float` - Upper limit of integration (must be in [0, 1])
- `alpha: float` - Shape parameter α > 0
- `beta: float` - Shape parameter β > 0
- `steps: int` - Number of integration steps (default: 1000)

**Returns**: `float` - Cumulative probability P(X ≤ x)

**Example**:
```python
# Probability that conversion rate ≤ 50% for Beta(2,2)
prob = mb.beta_cdf(0.5, 2, 2)  # Returns: 0.5

# 95th percentile calculation
prob_95 = mb.beta_cdf(0.95, 1, 1)  # Returns: 0.95
```

**Use Cases**:
- Confidence interval calculations
- Percentile estimation
- Hypothesis testing

---

### `beta_mean(alpha, beta)`

Calculate the mean (expected value) of the Beta distribution.

**Formula**: `E[X] = α / (α + β)`

**Parameters**:
- `alpha: float` - Shape parameter α > 0
- `beta: float` - Shape parameter β > 0

**Returns**: `float` - Mean of the distribution

**Example**:
```python
# Expected conversion rate for Beta(12, 88) posterior
mean_rate = mb.beta_mean(12, 88)  # Returns: 0.12 (12%)

# Uniform prior mean
uniform_mean = mb.beta_mean(1, 1)  # Returns: 0.5 (50%)
```

**Use Cases**:
- Point estimates in Bayesian analysis
- Expected value calculations
- Business metric estimation

---

### `beta_var(alpha, beta)`

Calculate the variance of the Beta distribution.

**Formula**: `Var[X] = (αβ) / ((α + β)² * (α + β + 1))`

**Parameters**:
- `alpha: float` - Shape parameter α > 0
- `beta: float` - Shape parameter β > 0

**Returns**: `float` - Variance of the distribution

**Example**:
```python
# Uncertainty in conversion rate estimate
variance = mb.beta_var(12, 88)  # Returns: ~0.00106
std_dev = variance ** 0.5       # Standard deviation: ~0.033

# High uncertainty with weak prior
high_var = mb.beta_var(1, 1)    # Returns: 0.083 (uniform)
```

**Use Cases**:
- Uncertainty quantification
- Confidence interval width estimation
- Risk assessment

---

### `beta_mode(alpha, beta)`

Calculate the mode (most likely value) of the Beta distribution.

**Formula**: `Mode = (α - 1) / (α + β - 2)` for α > 1, β > 1

**Parameters**:
- `alpha: float` - Shape parameter α > 0
- `beta: float` - Shape parameter β > 0

**Returns**: `float` - Mode of the distribution

**Example**:
```python
# Most likely conversion rate
mode = mb.beta_mode(12, 88)  # Returns: ~0.111 (11.1%)

# Bimodal case (α < 1 or β < 1)
try:
    mode = mb.beta_mode(0.5, 0.5)  # Raises ValueError
except ValueError:
    print("No unique mode for this distribution")
```

**Use Cases**:
- Maximum a posteriori (MAP) estimation
- Most likely outcome prediction
- Decision making under uncertainty

## Bayes' Theorem Functions

### `bayes_posterior(prior, likelihood, evidence)`

Calculate posterior probability using Bayes' theorem.

**Formula**: `P(H|E) = P(E|H) * P(H) / P(E)`

**Parameters**:
- `prior: float` - Prior probability P(H)
- `likelihood: float` - Likelihood P(E|H)  
- `evidence: float` - Evidence P(E)

**Returns**: `float` - Posterior probability P(H|E)

**Example**:
```python
# Medical diagnosis example
prior_disease = 0.01        # 1% disease prevalence
test_sensitivity = 0.95     # 95% true positive rate
evidence = 0.059           # P(positive test)

posterior = mb.bayes_posterior(prior_disease, test_sensitivity, evidence)
# Returns: ~0.161 (16.1% chance of disease given positive test)
```

**Use Cases**:
- Medical diagnosis
- Spam filtering
- Fraud detection

---

### `bayes_odds(prior_odds, likelihood_ratio)`

Calculate posterior odds using Bayes' factor.

**Formula**: `Posterior Odds = Prior Odds × Likelihood Ratio`

**Parameters**:
- `prior_odds: float` - Prior odds ratio
- `likelihood_ratio: float` - Bayes factor (likelihood ratio)

**Returns**: `float` - Posterior odds ratio

**Example**:
```python
# A/B testing with odds
prior_odds = 1.0           # Equal prior belief
likelihood_ratio = 2.5     # Evidence favors treatment 2.5:1

posterior_odds = mb.bayes_odds(prior_odds, likelihood_ratio)
# Returns: 2.5 (treatment 2.5x more likely to be better)
```

**Use Cases**:
- A/B testing analysis
- Model comparison
- Evidence evaluation

---

### `bayes_update_discrete(prior_probs, likelihoods)`

Update discrete probability distribution using Bayes' theorem.

**Formula**: `P(Hᵢ|E) = P(E|Hᵢ) * P(Hᵢ) / Σⱼ P(E|Hⱼ) * P(Hⱼ)`

**Parameters**:
- `prior_probs: List[float]` - Prior probabilities for each hypothesis
- `likelihoods: List[float]` - Likelihood of evidence under each hypothesis

**Returns**: `List[float]` - Posterior probabilities (normalized)

**Example**:
```python
# Multiple hypothesis testing
priors = [0.33, 0.33, 0.34]      # Three equally likely hypotheses
likelihoods = [0.8, 0.6, 0.2]    # Evidence strength for each

posteriors = mb.bayes_update_discrete(priors, likelihoods)
# Returns: [0.5, 0.375, 0.125] - first hypothesis most likely
```

**Use Cases**:
- Multi-class classification
- Hypothesis ranking
- Decision trees

---

### `marginal_likelihood_discrete(prior_probs, likelihoods)`

Calculate marginal likelihood (evidence) for discrete case.

**Formula**: `P(E) = Σᵢ P(E|Hᵢ) * P(Hᵢ)`

**Parameters**:
- `prior_probs: List[float]` - Prior probabilities
- `likelihoods: List[float]` - Likelihoods under each hypothesis

**Returns**: `float` - Marginal likelihood (evidence)

**Example**:
```python
# Calculate evidence for model comparison
priors = [0.5, 0.5]
likelihoods = [0.8, 0.3]

evidence = mb.marginal_likelihood_discrete(priors, likelihoods)
# Returns: 0.55 - total probability of observing the evidence
```

**Use Cases**:
- Model evidence calculation
- Bayes factor computation
- Model selection

## Conjugate Prior Functions

### `update_beta_binomial(alpha_prior, beta_prior, successes, failures)`

Update Beta prior with binomial data (Beta-Binomial conjugacy).

**Formula**: `Beta(α + s, β + f)` where s = successes, f = failures

**Parameters**:
- `alpha_prior: float` - Prior α parameter
- `beta_prior: float` - Prior β parameter  
- `successes: int` - Number of observed successes
- `failures: int` - Number of observed failures

**Returns**: `Tuple[float, float]` - Posterior (α, β) parameters

**Example**:
```python
# A/B testing: update conversion rate belief
alpha_post, beta_post = mb.update_beta_binomial(
    alpha_prior=1,    # Uniform prior
    beta_prior=1,
    successes=15,     # 15 conversions
    failures=85       # 85 non-conversions
)
# Returns: (16, 86) - posterior Beta(16, 86)

# Expected conversion rate
conversion_rate = mb.beta_mean(alpha_post, beta_post)  # ~15.7%
```

**Use Cases**:
- A/B testing
- Conversion rate optimization
- Quality control

---

### `update_normal_known_variance(mu_prior, sigma_prior, data_mean, data_variance, n)`

Update Normal prior with known variance (Normal-Normal conjugacy).

**Formula**: Complex conjugate update for Normal distribution with known variance

**Parameters**:
- `mu_prior: float` - Prior mean
- `sigma_prior: float` - Prior standard deviation
- `data_mean: float` - Sample mean
- `data_variance: float` - Known data variance
- `n: int` - Sample size

**Returns**: `Tuple[float, float]` - Posterior (μ, σ) parameters

**Example**:
```python
# Update belief about average customer spend
mu_post, sigma_post = mb.update_normal_known_variance(
    mu_prior=50.0,      # Prior: $50 average
    sigma_prior=10.0,   # Prior uncertainty: $10
    data_mean=55.0,     # Observed: $55 average
    data_variance=100.0, # Known variance: $100
    n=25               # Sample size: 25 customers
)
# Returns updated posterior parameters
```

**Use Cases**:
- Parameter estimation
- Quality control
- Financial modeling

---

### `update_poisson_gamma(alpha_prior, beta_prior, data_sum, n)`

Update Gamma prior with Poisson data (Gamma-Poisson conjugacy).

**Formula**: `Gamma(α + Σx, β + n)` where Σx = sum of observations, n = count

**Parameters**:
- `alpha_prior: float` - Prior shape parameter
- `beta_prior: float` - Prior rate parameter
- `data_sum: float` - Sum of observed counts
- `n: int` - Number of observations

**Returns**: `Tuple[float, float]` - Posterior (α, β) parameters

**Example**:
```python
# Update belief about website traffic rate
alpha_post, beta_post = mb.update_poisson_gamma(
    alpha_prior=2.0,    # Weak prior
    beta_prior=1.0,
    data_sum=150,       # Total visits observed
    n=7                # Over 7 days
)
# Returns: (152, 8) - posterior Gamma(152, 8)

# Expected rate
expected_rate = alpha_post / beta_post  # ~19 visits/day
```

**Use Cases**:
- Count data analysis
- Rate estimation
- Reliability engineering

## Utility Functions

### `gamma_func(x)`

Compute the Gamma function Γ(x).

**Formula**: `Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt`

**Parameters**:
- `x: float` - Input value (x > 0)

**Returns**: `float` - Gamma function value

**Example**:
```python
# Gamma function values
gamma_2 = mb.gamma_func(2)    # Returns: 1.0 (since Γ(2) = 1!)
gamma_half = mb.gamma_func(0.5)  # Returns: √π ≈ 1.772
```

---

### `validate_prob(x)`

Validate that a value is a valid probability.

**Parameters**:
- `x: float` - Value to validate

**Returns**: `bool` - True if valid probability (0 ≤ x ≤ 1)

---

### `normalize(probs)`

Normalize a list of values to sum to 1.

**Parameters**:
- `probs: List[float]` - Values to normalize

**Returns**: `List[float]` - Normalized probabilities

---

### `safe_div(a, b, default=0.0)`

Perform safe division with default value for division by zero.

**Parameters**:
- `a: float` - Numerator
- `b: float` - Denominator  
- `default: float` - Default value if b = 0

**Returns**: `float` - a/b or default if b = 0

## Complete Example: A/B Testing Pipeline

```python
import metricax.bayesian as mb

# A/B Testing Complete Workflow
def ab_test_analysis(control_conversions, control_visitors, 
                    treatment_conversions, treatment_visitors):
    
    # 1. Set up priors (uniform)
    alpha_prior, beta_prior = 1, 1
    
    # 2. Update posteriors with data
    control_alpha, control_beta = mb.update_beta_binomial(
        alpha_prior, beta_prior, 
        control_conversions, control_visitors - control_conversions
    )
    
    treatment_alpha, treatment_beta = mb.update_beta_binomial(
        alpha_prior, beta_prior,
        treatment_conversions, treatment_visitors - treatment_conversions  
    )
    
    # 3. Calculate key metrics
    control_rate = mb.beta_mean(control_alpha, control_beta)
    treatment_rate = mb.beta_mean(treatment_alpha, treatment_beta)
    
    control_var = mb.beta_var(control_alpha, control_beta)
    treatment_var = mb.beta_var(treatment_alpha, treatment_beta)
    
    # 4. Results
    lift = (treatment_rate - control_rate) / control_rate
    
    return {
        'control_rate': control_rate,
        'treatment_rate': treatment_rate,
        'lift': lift,
        'control_uncertainty': control_var ** 0.5,
        'treatment_uncertainty': treatment_var ** 0.5
    }

# Run analysis
results = ab_test_analysis(12, 120, 15, 100)
print(f"Control: {results['control_rate']:.1%}")
print(f"Treatment: {results['treatment_rate']:.1%}")  
print(f"Lift: {results['lift']:.1%}")
```
