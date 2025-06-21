# MetricaX Examples 🎯

This directory contains real-world examples demonstrating how to use MetricaX for practical Bayesian analysis tasks.

## 📊 Available Examples

### 1. A/B Testing Analysis (`ab_testing.py`)

**Scenario**: Compare conversion rates between two website variants

**What you'll learn**:
- How to use Beta-Binomial conjugate priors
- Statistical significance testing with Bayesian methods
- Calculating probability that one variant is better
- Making business decisions with uncertainty quantification

**Key Functions Used**:
- `update_beta_binomial()`
- `beta_mean()`, `beta_var()`
- `beta_pdf()`

**Run it**:
```bash
python -m metricax.bayesian.examples.ab_testing
```

**Sample Output**:
```
🧪 A/B Testing with Bayesian Analysis
==================================================
Variant A (Control): 12/120 conversions
Variant B (Treatment): 15/100 conversions

📊 Posterior Distributions:
Variant A: Beta(13.0, 109.0)
Variant B: Beta(16.0, 86.0)

🎯 Decision Analysis:
Probability that B > A: 0.876
⚠️  Moderate evidence that B is better than A
```

### 2. Bayesian Spam Filter (`spam_filter.py`)

**Scenario**: Build an email spam classifier using Bayesian inference

**What you'll learn**:
- Text classification with Bayes' theorem
- Handling multiple features (words) in classification
- Laplace smoothing for unseen data
- Converting between probability and odds forms

**Key Functions Used**:
- `bayes_update_discrete()`
- `bayes_posterior()`
- `bayes_odds()`

**Run it**:
```bash
python -m metricax.bayesian.examples.spam_filter
```

**Sample Output**:
```
📧 Bayesian Spam Filter Demo
========================================
📚 Training completed:
   Spam emails: 4
   Ham emails: 6
   Vocabulary size: 25

🧪 Classification Results:
Email 1: 'buy cheap pills online now'
   Prediction: SPAM
   P(Spam): 0.892
   P(Ham):  0.108
   Confidence: 0.892
```

### 3. Data Update Examples (`data_updates.py`)

**Scenario**: Multiple real-world scenarios showing online Bayesian learning

**What you'll learn**:
- Manufacturing quality control with Beta updates
- Sensor calibration with Normal updates
- Error rate monitoring with Poisson-Gamma updates
- How beliefs evolve as new data arrives

**Key Functions Used**:
- `update_beta_binomial()`
- `update_normal_known_variance()`
- `update_poisson_gamma()`

**Run it**:
```bash
python -m metricax.bayesian.examples.data_updates
```

**Sample Output**:
```
🏭 Manufacturing Quality Control with Bayesian Updates
=======================================================
📊 Initial belief: Beta(2, 98)
   Expected defect rate: 0.020

📦 Processing batches sequentially:
Batch 1: 3/100 defects
   Updated: Beta(5.0, 195.0)
   Defect rate: 0.0250 ± 0.0111
   ✅ Quality within acceptable range
```

## 🚀 Running the Examples

### Prerequisites

```bash
# Basic requirements (included with MetricaX)
pip install metricax

# Optional for visualizations
pip install matplotlib numpy
```

### Run Individual Examples

```bash
# Method 1: Direct execution
python -m metricax.bayesian.examples.ab_testing
python -m metricax.bayesian.examples.spam_filter
python -m metricax.bayesian.examples.data_updates

# Method 2: Import and run
python -c "from metricax.bayesian.examples import ab_testing; ab_testing.run_example()"
python -c "from metricax.bayesian.examples import spam_filter; spam_filter.run_example()"
python -c "from metricax.bayesian.examples import data_updates; data_updates.run_example()"
```

### Run All Examples

```bash
# From Python
python -c "
from metricax.bayesian.examples import ab_testing, spam_filter, data_updates
ab_testing.run_example()
spam_filter.run_example()
data_updates.run_example()
"
```

## 🎓 Learning Path

**Beginner**: Start with `ab_testing_example.py`
- Simple scenario with clear business impact
- Introduces Beta distributions and conjugate priors
- Shows practical decision making

**Intermediate**: Try `spam_filter_example.py`
- More complex with multiple features
- Demonstrates discrete Bayesian updates
- Real classification problem

**Advanced**: Explore `data_update_example.py`
- Multiple conjugate prior families
- Online learning scenarios
- Industrial applications

## 🔧 Customizing Examples

Each example is designed to be easily modified:

### A/B Testing
- Change conversion rates and sample sizes
- Add more variants (A/B/C testing)
- Modify prior beliefs

### Spam Filter
- Add more training data
- Include additional features
- Experiment with different priors

### Data Updates
- Modify quality thresholds
- Change update frequencies
- Add new monitoring scenarios

## 📚 Mathematical Background

### Beta-Binomial Conjugacy
```
Prior: Beta(α, β)
Likelihood: Binomial(n, p) with k successes
Posterior: Beta(α + k, β + n - k)
```

### Normal-Normal Conjugacy
```
Prior: N(μ₀, σ₀²)
Likelihood: N(μ, σ²) with known σ²
Posterior: N(μₙ, σₙ²) - computed analytically
```

### Gamma-Poisson Conjugacy
```
Prior: Gamma(α, β)
Likelihood: Poisson(λ) with observations
Posterior: Gamma(α + Σx, β + n)
```

## 🤝 Contributing Examples

Have a great use case? We'd love to include it!

1. Create a new `.py` file following the existing pattern
2. Include comprehensive docstrings and comments
3. Add sample output in this README
4. Test that it runs without errors
5. Submit a pull request

## 📞 Support

If you have questions about the examples:
- Check the inline comments in each script
- Review the main MetricaX documentation
- Open an issue on GitHub

---

**Happy Bayesian modeling! 🎉**
