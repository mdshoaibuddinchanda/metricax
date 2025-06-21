#!/usr/bin/env python3
"""
A/B Testing with Bayesian Analysis using MetricaX

This example demonstrates how to use MetricaX for A/B testing analysis,
comparing conversion rates between two website variants using Beta distributions.
"""

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import from parent bayesian module
from .. import (
    update_beta_binomial, beta_mean, beta_var, beta_pdf
)


def ab_test_analysis():
    """
    Real-world A/B testing scenario:
    - Variant A (Control): 120 visitors, 12 conversions
    - Variant B (Treatment): 100 visitors, 15 conversions
    """
    
    print("🧪 A/B Testing with Bayesian Analysis")
    print("=" * 50)
    
    # Data from our A/B test
    visitors_a, conversions_a = 120, 12
    visitors_b, conversions_b = 100, 15
    
    print(f"Variant A (Control): {conversions_a}/{visitors_a} conversions")
    print(f"Variant B (Treatment): {conversions_b}/{visitors_b} conversions")
    print()
    
    # Prior beliefs (uninformative prior: Beta(1,1) = Uniform)
    prior_alpha, prior_beta = 1, 1
    
    # Update beliefs with observed data using conjugate priors
    alpha_a, beta_a = update_beta_binomial(
        prior_alpha, prior_beta, 
        conversions_a, visitors_a - conversions_a
    )
    
    alpha_b, beta_b = update_beta_binomial(
        prior_alpha, prior_beta,
        conversions_b, visitors_b - conversions_b
    )
    
    print("📊 Posterior Distributions:")
    print(f"Variant A: Beta({alpha_a}, {beta_a})")
    print(f"Variant B: Beta({alpha_b}, {beta_b})")
    print()
    
    # Calculate posterior statistics
    mean_a = beta_mean(alpha_a, beta_a)
    mean_b = beta_mean(alpha_b, beta_b)
    var_a = beta_var(alpha_a, beta_a)
    var_b = beta_var(alpha_b, beta_b)
    
    print("📈 Conversion Rate Estimates:")
    if HAS_NUMPY:
        print(f"Variant A: {mean_a:.3f} ± {np.sqrt(var_a):.3f}")
        print(f"Variant B: {mean_b:.3f} ± {np.sqrt(var_b):.3f}")
    else:
        print(f"Variant A: {mean_a:.3f} ± {var_a**0.5:.3f}")
        print(f"Variant B: {mean_b:.3f} ± {var_b**0.5:.3f}")
    print()
    
    # Calculate probability that B > A
    if HAS_NUMPY:
        # Using Monte Carlo simulation
        n_samples = 10000
        samples_a = np.random.beta(alpha_a, beta_a, n_samples)
        samples_b = np.random.beta(alpha_b, beta_b, n_samples)
        prob_b_better = np.mean(samples_b > samples_a)
    else:
        # Simple approximation using normal approximation
        import math
        z_score = (mean_b - mean_a) / math.sqrt(var_a + var_b)
        # Rough approximation of P(Z > -z_score)
        prob_b_better = 0.5 + 0.5 * math.erf(z_score / math.sqrt(2))
    
    print("🎯 Decision Analysis:")
    print(f"Probability that B > A: {prob_b_better:.3f}")
    
    if prob_b_better > 0.95:
        print("✅ Strong evidence that B is better than A")
    elif prob_b_better > 0.8:
        print("⚠️  Moderate evidence that B is better than A")
    elif prob_b_better < 0.2:
        print("❌ Strong evidence that A is better than B")
    else:
        print("🤔 Inconclusive - need more data")
    
    print()
    print("💡 Business Recommendation:")
    if prob_b_better > 0.8:
        lift = (mean_b - mean_a) / mean_a * 100
        print(f"   Implement Variant B (estimated {lift:+.1f}% lift)")
    else:
        print("   Continue testing or stick with Variant A")
    
    return {
        'variant_a': {'alpha': alpha_a, 'beta': beta_a, 'mean': mean_a},
        'variant_b': {'alpha': alpha_b, 'beta': beta_b, 'mean': mean_b},
        'prob_b_better': prob_b_better
    }


def plot_distributions():
    """Plot the posterior distributions for visualization"""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("📊 Install matplotlib and numpy to see the visualization:")
        print("   pip install matplotlib numpy")
        return
    
    # Posterior parameters from the analysis above
    alpha_a, beta_a = 13, 109  # 1 + 12, 1 + (120-12)
    alpha_b, beta_b = 16, 86   # 1 + 15, 1 + (100-15)
    
    x = np.linspace(0, 0.3, 1000)
    
    # Calculate PDFs
    pdf_a = [beta_pdf(xi, alpha_a, beta_a) for xi in x]
    pdf_b = [beta_pdf(xi, alpha_b, beta_b) for xi in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf_a, label='Variant A (Control)', linewidth=2)
    plt.plot(x, pdf_b, label='Variant B (Treatment)', linewidth=2)
    plt.axvline(beta_mean(alpha_a, beta_a), color='blue', linestyle='--', alpha=0.7)
    plt.axvline(beta_mean(alpha_b, beta_b), color='orange', linestyle='--', alpha=0.7)
    
    plt.xlabel('Conversion Rate')
    plt.ylabel('Probability Density')
    plt.title('Posterior Distributions of Conversion Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_example():
    """Main function to run the A/B testing example"""
    results = ab_test_analysis()
    print("\n" + "="*50)
    plot_distributions()
    return results


if __name__ == "__main__":
    run_example()
