"""
Entropy Analysis Examples

Demonstrates practical applications of entropy measures for:
- Feature selection in machine learning
- Model comparison and evaluation
- Uncertainty quantification
- Cross-entropy loss analysis

@formula: H(X) = -∑ p(x) log p(x)
@source: Cover & Thomas, Elements of Information Theory
"""

import random
from typing import List, Tuple, Dict
import metricax.info_theory as it


def feature_selection_example() -> Dict[str, float]:
    """
    Demonstrate feature selection using entropy measures.
    
    Simulates a binary classification problem where we rank features
    by their information content about the target variable.
    
    Returns:
        Dictionary with feature rankings and entropy measures
    """
    print("🎯 Feature Selection with Information Theory")
    print("=" * 50)
    
    # Simulate feature data (4 features, binary classification)
    # Feature 1: Highly informative (low entropy when conditioned on target)
    # Feature 2: Moderately informative  
    # Feature 3: Weakly informative
    # Feature 4: Random noise (high entropy)
    
    # Target distribution (balanced classes)
    target_dist = [0.5, 0.5]  # 50% class 0, 50% class 1
    target_entropy = it.entropy(target_dist)
    
    print(f"Target Entropy: {target_entropy:.3f} bits")
    print(f"Maximum possible entropy: {it.entropy([0.5, 0.5]):.3f} bits")
    print()
    
    # Feature distributions given target classes
    features = {
        "Feature_1_Informative": {
            "class_0": [0.9, 0.1],  # Strong signal
            "class_1": [0.1, 0.9],
        },
        "Feature_2_Moderate": {
            "class_0": [0.7, 0.3],  # Moderate signal
            "class_1": [0.3, 0.7],
        },
        "Feature_3_Weak": {
            "class_0": [0.6, 0.4],  # Weak signal
            "class_1": [0.4, 0.6],
        },
        "Feature_4_Noise": {
            "class_0": [0.5, 0.5],  # No signal (random)
            "class_1": [0.5, 0.5],
        }
    }
    
    results = {}
    
    for feature_name, distributions in features.items():
        # Calculate conditional entropy H(Feature|Target)
        conditional_entropy = (
            0.5 * it.entropy(distributions["class_0"]) +
            0.5 * it.entropy(distributions["class_1"])
        )
        
        # Calculate mutual information I(Feature; Target)
        feature_entropy = it.entropy([0.5, 0.5])  # Marginal entropy
        mutual_info = feature_entropy - conditional_entropy
        
        # Information gain (same as mutual information)
        info_gain = target_entropy - conditional_entropy
        
        results[feature_name] = {
            "entropy": feature_entropy,
            "conditional_entropy": conditional_entropy,
            "mutual_information": mutual_info,
            "information_gain": info_gain,
        }
        
        print(f"{feature_name}:")
        print(f"  Entropy: {feature_entropy:.3f} bits")
        print(f"  Conditional Entropy: {conditional_entropy:.3f} bits")
        print(f"  Mutual Information: {mutual_info:.3f} bits")
        print(f"  Information Gain: {info_gain:.3f} bits")
        print()
    
    # Rank features by information gain
    ranked_features = sorted(
        results.items(), 
        key=lambda x: x[1]["information_gain"], 
        reverse=True
    )
    
    print("🏆 Feature Ranking (by Information Gain):")
    for i, (feature, metrics) in enumerate(ranked_features, 1):
        print(f"{i}. {feature}: {metrics['information_gain']:.3f} bits")
    
    return results


def model_comparison_example() -> Dict[str, float]:
    """
    Compare machine learning models using cross-entropy and KL divergence.
    
    Simulates model predictions vs true distribution to demonstrate
    how information theory measures model quality.
    
    Returns:
        Dictionary with model comparison metrics
    """
    print("\n🤖 Model Comparison with Cross-Entropy")
    print("=" * 50)
    
    # True distribution (ground truth)
    true_dist = [0.7, 0.2, 0.1]  # 3-class problem
    
    # Model predictions (different quality levels)
    models = {
        "Perfect_Model": [0.7, 0.2, 0.1],      # Matches true distribution
        "Good_Model": [0.65, 0.25, 0.1],       # Close to true distribution  
        "Poor_Model": [0.4, 0.4, 0.2],         # Far from true distribution
        "Overconfident_Model": [0.9, 0.05, 0.05],  # Overconfident
        "Uniform_Model": [0.33, 0.33, 0.34],   # Uniform (maximum entropy)
    }
    
    results = {}
    
    print(f"True Distribution: {true_dist}")
    print()
    
    for model_name, pred_dist in models.items():
        # Cross-entropy H(true, predicted)
        cross_ent = it.cross_entropy(true_dist, pred_dist)
        
        # KL divergence D(true || predicted)
        kl_div = it.kl_divergence(true_dist, pred_dist)
        
        # Model entropy (prediction confidence)
        model_entropy = it.entropy(pred_dist)
        
        results[model_name] = {
            "cross_entropy": cross_ent,
            "kl_divergence": kl_div,
            "model_entropy": model_entropy,
            "predictions": pred_dist,
        }
        
        print(f"{model_name}:")
        print(f"  Predictions: {pred_dist}")
        print(f"  Cross-Entropy: {cross_ent:.3f} bits")
        print(f"  KL Divergence: {kl_div:.3f} bits")
        print(f"  Model Entropy: {model_entropy:.3f} bits")
        print()
    
    # Rank models by cross-entropy (lower is better)
    ranked_models = sorted(
        results.items(),
        key=lambda x: x[1]["cross_entropy"]
    )
    
    print("🏆 Model Ranking (by Cross-Entropy, lower is better):")
    for i, (model, metrics) in enumerate(ranked_models, 1):
        print(f"{i}. {model}: {metrics['cross_entropy']:.3f} bits")
    
    return results


def uncertainty_quantification_example() -> Dict[str, float]:
    """
    Demonstrate uncertainty quantification using entropy measures.
    
    Shows how entropy can measure prediction uncertainty in different scenarios.
    
    Returns:
        Dictionary with uncertainty analysis results
    """
    print("\n🎲 Uncertainty Quantification with Entropy")
    print("=" * 50)
    
    scenarios = {
        "Certain_Prediction": [0.95, 0.03, 0.02],      # Very confident
        "Moderate_Uncertainty": [0.6, 0.3, 0.1],       # Moderate confidence
        "High_Uncertainty": [0.4, 0.35, 0.25],         # Low confidence
        "Maximum_Uncertainty": [0.33, 0.33, 0.34],     # Uniform (max entropy)
        "Binary_Certain": [0.99, 0.01],                # Binary, very certain
        "Binary_Uncertain": [0.51, 0.49],              # Binary, uncertain
    }
    
    results = {}
    max_entropy_3class = it.entropy([1/3, 1/3, 1/3])
    max_entropy_2class = it.entropy([0.5, 0.5])
    
    print(f"Maximum Entropy (3-class): {max_entropy_3class:.3f} bits")
    print(f"Maximum Entropy (2-class): {max_entropy_2class:.3f} bits")
    print()
    
    for scenario_name, distribution in scenarios.items():
        ent = it.entropy(distribution)
        
        # Normalized uncertainty (0 = certain, 1 = maximum uncertainty)
        if len(distribution) == 3:
            normalized_uncertainty = ent / max_entropy_3class
        else:
            normalized_uncertainty = ent / max_entropy_2class
        
        results[scenario_name] = {
            "entropy": ent,
            "normalized_uncertainty": normalized_uncertainty,
            "distribution": distribution,
        }
        
        print(f"{scenario_name}:")
        print(f"  Distribution: {distribution}")
        print(f"  Entropy: {ent:.3f} bits")
        print(f"  Normalized Uncertainty: {normalized_uncertainty:.1%}")
        print()
    
    return results


def run_example() -> None:
    """
    Run all entropy analysis examples.
    
    Demonstrates practical applications of entropy measures in:
    - Feature selection for machine learning
    - Model comparison and evaluation  
    - Uncertainty quantification
    """
    print("🔬 MetricaX Information Theory - Entropy Analysis Examples")
    print("=" * 60)
    
    # Run all examples
    feature_results = feature_selection_example()
    model_results = model_comparison_example()
    uncertainty_results = uncertainty_quantification_example()
    
    print("\n✅ All examples completed successfully!")
    print("\n💡 Key Takeaways:")
    print("• Higher information gain = better features for classification")
    print("• Lower cross-entropy = better model predictions")
    print("• Higher entropy = more uncertainty in predictions")
    print("• Information theory provides objective measures for ML evaluation")


if __name__ == "__main__":
    run_example()
