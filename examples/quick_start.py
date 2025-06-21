"""
MetricaX Quick Start Guide

This file demonstrates the core capabilities of MetricaX with practical examples
that you can run immediately after installation.

Run this file to see MetricaX in action:
    python examples/quick_start.py

Topics covered:
1. Bayesian A/B Testing
2. Information Theory for Feature Selection
3. Distribution Analysis
4. Real-world Applications
"""

import metricax.bayesian as mb
import metricax.info_theory as it


def bayesian_ab_testing_demo():
    """
    Demonstrate Bayesian A/B testing with conversion rate analysis.
    
    Scenario: E-commerce website testing two checkout button designs
    """
    print("🎯 Bayesian A/B Testing Demo")
    print("=" * 40)
    print("Scenario: Testing two checkout button designs")
    print()
    
    # Test data
    control_conversions = 12
    control_visitors = 120
    treatment_conversions = 18
    treatment_visitors = 100
    
    print(f"Control: {control_conversions}/{control_visitors} conversions")
    print(f"Treatment: {treatment_conversions}/{treatment_visitors} conversions")
    print()
    
    # Bayesian analysis with Beta-Binomial conjugate priors
    # Start with uniform prior Beta(1,1)
    
    # Update posteriors with observed data
    control_alpha, control_beta = mb.update_beta_binomial(
        1, 1,  # Uniform prior
        control_conversions, 
        control_visitors - control_conversions
    )
    
    treatment_alpha, treatment_beta = mb.update_beta_binomial(
        1, 1,  # Uniform prior  
        treatment_conversions,
        treatment_visitors - treatment_conversions
    )
    
    # Calculate posterior statistics
    control_rate = mb.beta_mean(control_alpha, control_beta)
    treatment_rate = mb.beta_mean(treatment_alpha, treatment_beta)
    
    control_std = mb.beta_var(control_alpha, control_beta) ** 0.5
    treatment_std = mb.beta_var(treatment_alpha, treatment_beta) ** 0.5
    
    # Business metrics
    relative_lift = (treatment_rate - control_rate) / control_rate
    absolute_lift = treatment_rate - control_rate
    
    print("📊 Results:")
    print(f"Control Rate: {control_rate:.1%} ± {control_std:.1%}")
    print(f"Treatment Rate: {treatment_rate:.1%} ± {treatment_std:.1%}")
    print(f"Absolute Lift: {absolute_lift:+.1%}")
    print(f"Relative Lift: {relative_lift:+.1%}")
    print()
    
    # Confidence assessment
    if treatment_rate > control_rate:
        print("✅ Treatment appears better than control")
        confidence = "High" if treatment_std < 0.02 else "Moderate"
        print(f"Confidence Level: {confidence}")
    else:
        print("❌ Treatment does not outperform control")
    
    print()
    return {
        'control_rate': control_rate,
        'treatment_rate': treatment_rate,
        'lift': relative_lift
    }


def information_theory_feature_selection_demo():
    """
    Demonstrate feature selection using information theory measures.
    
    Scenario: Selecting features for customer churn prediction
    """
    print("🔍 Information Theory Feature Selection Demo")
    print("=" * 50)
    print("Scenario: Customer churn prediction feature ranking")
    print()
    
    # Simulate customer data distributions
    # Target: churn (0=stay, 1=churn)
    target_dist = [0.8, 0.2]  # 20% churn rate
    target_entropy = it.entropy(target_dist)
    
    print(f"Target Distribution (Stay/Churn): {target_dist}")
    print(f"Target Entropy: {target_entropy:.3f} bits")
    print()
    
    # Feature analysis
    features = {
        "Account_Age": {
            "description": "Customer account age (months)",
            "distribution": [0.3, 0.4, 0.3],  # New, Medium, Old
            "churn_correlation": "High"  # Newer customers churn more
        },
        "Support_Tickets": {
            "description": "Number of support tickets",
            "distribution": [0.6, 0.3, 0.1],  # Low, Medium, High
            "churn_correlation": "High"  # More tickets = more churn
        },
        "Monthly_Spend": {
            "description": "Average monthly spending",
            "distribution": [0.4, 0.4, 0.2],  # Low, Medium, High
            "churn_correlation": "Medium"  # Some correlation
        },
        "Login_Frequency": {
            "description": "Login frequency per week",
            "distribution": [0.33, 0.33, 0.34],  # Low, Medium, High
            "churn_correlation": "Low"  # Weak correlation
        }
    }
    
    # Calculate information gain for each feature
    feature_scores = {}
    
    for feature_name, feature_data in features.items():
        feature_dist = feature_data["distribution"]
        feature_entropy = it.entropy(feature_dist)
        
        # Simulate conditional entropy based on correlation strength
        correlation = feature_data["churn_correlation"]
        if correlation == "High":
            conditional_entropy = target_entropy * 0.3  # Low conditional entropy
        elif correlation == "Medium":
            conditional_entropy = target_entropy * 0.6  # Medium conditional entropy
        else:
            conditional_entropy = target_entropy * 0.9  # High conditional entropy
        
        # Information gain = H(Target) - H(Target|Feature)
        information_gain = target_entropy - conditional_entropy
        
        # Mutual information (same as information gain in this case)
        mutual_info = information_gain
        
        feature_scores[feature_name] = {
            "entropy": feature_entropy,
            "information_gain": information_gain,
            "mutual_information": mutual_info,
            "description": feature_data["description"]
        }
        
        print(f"{feature_name}:")
        print(f"  Description: {feature_data['description']}")
        print(f"  Feature Entropy: {feature_entropy:.3f} bits")
        print(f"  Information Gain: {information_gain:.3f} bits")
        print(f"  Mutual Information: {mutual_info:.3f} bits")
        print()
    
    # Rank features by information gain
    ranked_features = sorted(
        feature_scores.items(),
        key=lambda x: x[1]["information_gain"],
        reverse=True
    )
    
    print("🏆 Feature Ranking (by Information Gain):")
    for i, (feature, scores) in enumerate(ranked_features, 1):
        print(f"{i}. {feature}: {scores['information_gain']:.3f} bits")
    
    print()
    print("💡 Recommendation:")
    top_features = [name for name, _ in ranked_features[:2]]
    print(f"Use top 2 features for modeling: {', '.join(top_features)}")
    
    print()
    return ranked_features


def distribution_analysis_demo():
    """
    Demonstrate distribution analysis and comparison.
    
    Scenario: Comparing user behavior patterns across different app versions
    """
    print("📈 Distribution Analysis Demo")
    print("=" * 35)
    print("Scenario: Comparing user session lengths across app versions")
    print()
    
    # Session length distributions (discretized into bins)
    # Bins: <5min, 5-15min, 15-30min, 30-60min, >60min
    
    distributions = {
        "Version_1.0": [0.4, 0.3, 0.2, 0.08, 0.02],  # Shorter sessions
        "Version_2.0": [0.2, 0.3, 0.3, 0.15, 0.05],  # Longer sessions
        "Version_2.1": [0.25, 0.35, 0.25, 0.12, 0.03]  # Mixed pattern
    }
    
    print("Session Length Distributions:")
    bins = ["<5min", "5-15min", "15-30min", "30-60min", ">60min"]
    
    for version, dist in distributions.items():
        print(f"{version}: {[f'{p:.1%}' for p in dist]}")
    
    print()
    
    # Calculate entropy for each distribution
    print("📊 Distribution Analysis:")
    entropies = {}
    
    for version, dist in distributions.items():
        entropy = it.entropy(dist)
        entropies[version] = entropy
        
        # Interpret entropy
        max_entropy = it.entropy([0.2, 0.2, 0.2, 0.2, 0.2])  # Uniform
        uncertainty_level = entropy / max_entropy
        
        if uncertainty_level > 0.9:
            interpretation = "High uncertainty (diverse usage)"
        elif uncertainty_level > 0.7:
            interpretation = "Moderate uncertainty"
        else:
            interpretation = "Low uncertainty (concentrated usage)"
        
        print(f"{version}:")
        print(f"  Entropy: {entropy:.3f} bits")
        print(f"  Uncertainty Level: {uncertainty_level:.1%}")
        print(f"  Interpretation: {interpretation}")
        print()
    
    # Compare distributions using KL divergence
    print("🔄 Distribution Comparisons (KL Divergence):")
    
    base_version = "Version_1.0"
    base_dist = distributions[base_version]
    
    for version, dist in distributions.items():
        if version != base_version:
            kl_div = it.kl_divergence(base_dist, dist)
            
            # Interpret KL divergence
            if kl_div < 0.1:
                similarity = "Very similar"
            elif kl_div < 0.5:
                similarity = "Moderately similar"
            else:
                similarity = "Very different"
            
            print(f"{base_version} → {version}:")
            print(f"  KL Divergence: {kl_div:.3f} bits")
            print(f"  Similarity: {similarity}")
            print()
    
    return entropies


def real_world_applications_demo():
    """
    Showcase real-world applications combining Bayesian and Information Theory.
    
    Scenario: Email spam filter optimization
    """
    print("🛡️ Real-World Application: Spam Filter Optimization")
    print("=" * 55)
    print("Scenario: Optimizing email spam detection using Bayesian methods")
    print()
    
    # Email classification data
    # Features: contains_money_words, external_links, caps_ratio
    
    # Prior probabilities
    spam_rate = 0.3  # 30% of emails are spam
    ham_rate = 0.7   # 70% are legitimate
    
    print(f"Base Rates: {spam_rate:.1%} spam, {ham_rate:.1%} legitimate")
    print()
    
    # Feature likelihoods
    features = {
        "contains_money_words": {
            "spam_likelihood": 0.8,    # 80% of spam contains money words
            "ham_likelihood": 0.1,     # 10% of ham contains money words
        },
        "many_external_links": {
            "spam_likelihood": 0.7,    # 70% of spam has many links
            "ham_likelihood": 0.2,     # 20% of ham has many links
        },
        "high_caps_ratio": {
            "spam_likelihood": 0.6,    # 60% of spam is mostly caps
            "ham_likelihood": 0.05,    # 5% of ham is mostly caps
        }
    }
    
    # Test email with all three features present
    print("📧 Test Email Analysis:")
    print("Features detected: Money words, Many links, High caps ratio")
    print()
    
    # Calculate posterior probability using naive Bayes assumption
    # P(spam|features) ∝ P(spam) × ∏ P(feature|spam)
    
    spam_posterior = spam_rate
    ham_posterior = ham_rate
    
    print("Bayesian Update Process:")
    print(f"Initial: P(spam) = {spam_posterior:.3f}")
    
    for feature_name, likelihoods in features.items():
        # Update using Bayes' theorem
        spam_likelihood = likelihoods["spam_likelihood"]
        ham_likelihood = likelihoods["ham_likelihood"]
        
        # Calculate evidence (marginal likelihood)
        evidence = (spam_posterior * spam_likelihood + 
                   ham_posterior * ham_likelihood)
        
        # Update posteriors
        spam_posterior = (spam_likelihood * spam_posterior) / evidence
        ham_posterior = (ham_likelihood * ham_posterior) / evidence
        
        print(f"After {feature_name}: P(spam) = {spam_posterior:.3f}")
    
    print()
    print("📊 Final Classification:")
    print(f"P(Spam | Features) = {spam_posterior:.1%}")
    print(f"P(Ham | Features) = {ham_posterior:.1%}")
    
    # Decision
    if spam_posterior > 0.5:
        decision = "SPAM"
        confidence = spam_posterior
    else:
        decision = "HAM"
        confidence = ham_posterior
    
    print(f"Decision: {decision} (confidence: {confidence:.1%})")
    
    # Information theory analysis
    print()
    print("🔍 Information Theory Analysis:")
    
    # Calculate information gain for each feature
    # H(Class) - H(Class|Feature)
    
    class_entropy = it.entropy([spam_rate, ham_rate])
    print(f"Class Entropy: {class_entropy:.3f} bits")
    
    for feature_name, likelihoods in features.items():
        spam_like = likelihoods["spam_likelihood"]
        ham_like = likelihoods["ham_likelihood"]
        
        # Feature presence probability
        feature_prob = spam_rate * spam_like + ham_rate * ham_like
        
        # Conditional entropy H(Class|Feature=present)
        if feature_prob > 0:
            spam_given_feature = (spam_rate * spam_like) / feature_prob
            ham_given_feature = (ham_rate * ham_like) / feature_prob
            
            conditional_entropy = it.entropy([spam_given_feature, ham_given_feature])
            information_gain = class_entropy - conditional_entropy
            
            print(f"{feature_name}:")
            print(f"  Information Gain: {information_gain:.3f} bits")
            print(f"  Feature Value: {'High' if information_gain > 0.2 else 'Medium' if information_gain > 0.1 else 'Low'}")
    
    print()
    return {
        'spam_probability': spam_posterior,
        'decision': decision,
        'confidence': confidence
    }


def main():
    """
    Run all MetricaX quick start demonstrations.
    """
    print("🚀 MetricaX Quick Start Guide")
    print("=" * 50)
    print("Welcome to MetricaX - Professional Mathematical Toolkit")
    print("This demo showcases core capabilities with practical examples.")
    print()
    
    # Run all demonstrations
    ab_results = bayesian_ab_testing_demo()
    print("\n" + "="*60 + "\n")
    
    feature_results = information_theory_feature_selection_demo()
    print("\n" + "="*60 + "\n")
    
    distribution_results = distribution_analysis_demo()
    print("\n" + "="*60 + "\n")
    
    spam_results = real_world_applications_demo()
    print("\n" + "="*60 + "\n")
    
    # Summary
    print("✅ Quick Start Complete!")
    print()
    print("🎯 What you've learned:")
    print("• Bayesian A/B testing with conjugate priors")
    print("• Feature selection using information theory")
    print("• Distribution analysis and comparison")
    print("• Real-world spam filter optimization")
    print()
    print("📚 Next Steps:")
    print("• Explore detailed examples in metricax/*/examples/")
    print("• Read the API documentation in docs/")
    print("• Try the interactive Jupyter notebooks")
    print("• Check out the comprehensive test suite")
    print()
    print("🔗 Resources:")
    print("• GitHub: https://github.com/metricax/metricax")
    print("• Documentation: https://metricax.readthedocs.io")
    print("• Examples: ./examples/ and ./notebooks/")


if __name__ == "__main__":
    main()
