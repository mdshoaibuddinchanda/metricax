"""
Mutual Information Analysis Examples

Demonstrates practical applications of mutual information for:
- Feature dependency analysis
- Variable selection in high-dimensional data
- Network analysis and correlation detection
- Time series dependency analysis

@formula: I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
@source: Cover & Thomas, Elements of Information Theory
"""

import random
from typing import List, Tuple, Dict
import metricax.info_theory as it


def feature_dependency_example() -> Dict[str, float]:
    """
    Analyze dependencies between features using mutual information.
    
    Simulates a dataset with various types of feature relationships:
    - Independent features
    - Linearly dependent features  
    - Non-linearly dependent features
    - Redundant features
    
    Returns:
        Dictionary with dependency analysis results
    """
    print("🔗 Feature Dependency Analysis with Mutual Information")
    print("=" * 55)
    
    # Simulate different types of feature relationships
    # Each feature is discretized into 3 bins for simplicity
    
    features = {
        "Independent_A": [0.4, 0.3, 0.3],
        "Independent_B": [0.5, 0.2, 0.3],
        "Dependent_C": [0.6, 0.2, 0.2],    # Will be made dependent on A
        "Redundant_D": [0.4, 0.3, 0.3],    # Copy of A (maximum dependency)
        "Target": [0.3, 0.4, 0.3],
    }
    
    # Create joint distributions to simulate dependencies
    joint_distributions = {}
    
    # Independent A and B
    joint_AB = []
    for i, p_a in enumerate(features["Independent_A"]):
        for j, p_b in enumerate(features["Independent_B"]):
            joint_AB.append(p_a * p_b)  # Independence: P(A,B) = P(A)P(B)
    
    # Dependent A and C (moderate correlation)
    joint_AC = [
        0.3, 0.05, 0.05,  # A=0: mostly C=0
        0.05, 0.2, 0.05,  # A=1: mostly C=1  
        0.05, 0.05, 0.2   # A=2: mostly C=2
    ]
    
    # Redundant A and D (perfect correlation)
    joint_AD = [
        0.4, 0.0, 0.0,    # A=0 → D=0 always
        0.0, 0.3, 0.0,    # A=1 → D=1 always
        0.0, 0.0, 0.3     # A=2 → D=2 always
    ]
    
    joint_distributions = {
        ("Independent_A", "Independent_B"): joint_AB,
        ("Independent_A", "Dependent_C"): joint_AC,
        ("Independent_A", "Redundant_D"): joint_AD,
    }
    
    results = {}
    
    for (feat1, feat2), joint_dist in joint_distributions.items():
        # Calculate mutual information
        mutual_info = it.mutual_information(joint_dist)
        
        # Calculate individual entropies
        entropy_1 = it.entropy(features[feat1])
        entropy_2 = it.entropy(features[feat2])
        
        # Calculate joint entropy
        joint_entropy = it.entropy(joint_dist)
        
        # Verify: I(X;Y) = H(X) + H(Y) - H(X,Y)
        mutual_info_check = entropy_1 + entropy_2 - joint_entropy
        
        # Normalized mutual information (0 = independent, 1 = perfectly dependent)
        normalized_mi = mutual_info / min(entropy_1, entropy_2) if min(entropy_1, entropy_2) > 0 else 0
        
        results[f"{feat1}_vs_{feat2}"] = {
            "mutual_information": mutual_info,
            "normalized_mi": normalized_mi,
            "entropy_1": entropy_1,
            "entropy_2": entropy_2,
            "joint_entropy": joint_entropy,
            "mi_verification": mutual_info_check,
        }
        
        print(f"{feat1} vs {feat2}:")
        print(f"  Mutual Information: {mutual_info:.3f} bits")
        print(f"  Normalized MI: {normalized_mi:.3f} (0=independent, 1=dependent)")
        print(f"  H({feat1}): {entropy_1:.3f} bits")
        print(f"  H({feat2}): {entropy_2:.3f} bits")
        print(f"  H({feat1},{feat2}): {joint_entropy:.3f} bits")
        print(f"  Verification: {abs(mutual_info - mutual_info_check):.6f} (should be ~0)")
        print()
    
    return results


def feature_selection_ranking_example() -> Dict[str, float]:
    """
    Rank features for machine learning using mutual information with target.
    
    Demonstrates how to select the most informative features for prediction
    by measuring their mutual information with the target variable.
    
    Returns:
        Dictionary with feature rankings
    """
    print("🎯 Feature Selection Ranking with Mutual Information")
    print("=" * 55)
    
    # Target variable (3-class classification)
    target_dist = [0.4, 0.35, 0.25]
    
    # Features with different levels of informativeness about target
    features_vs_target = {
        "Highly_Informative": {
            # Strong relationship with target
            "joint": [
                0.35, 0.03, 0.02,  # Target=0: mostly Feature=0
                0.03, 0.30, 0.02,  # Target=1: mostly Feature=1
                0.02, 0.02, 0.21   # Target=2: mostly Feature=2
            ],
            "marginal": [0.4, 0.35, 0.25]
        },
        "Moderately_Informative": {
            # Moderate relationship with target
            "joint": [
                0.25, 0.10, 0.05,  # Target=0: prefers Feature=0
                0.10, 0.20, 0.05,  # Target=1: prefers Feature=1
                0.05, 0.05, 0.15   # Target=2: prefers Feature=2
            ],
            "marginal": [0.4, 0.35, 0.25]
        },
        "Weakly_Informative": {
            # Weak relationship with target
            "joint": [
                0.15, 0.13, 0.12,  # Target=0: slight preference for Feature=0
                0.13, 0.12, 0.10,  # Target=1: slight preference for Feature=1
                0.12, 0.10, 0.03   # Target=2: slight preference for Feature=2
            ],
            "marginal": [0.4, 0.35, 0.25]
        },
        "Non_Informative": {
            # No relationship with target (independent)
            "joint": [
                0.13, 0.14, 0.13,  # Target=0: uniform over features
                0.12, 0.12, 0.11,  # Target=1: uniform over features
                0.08, 0.09, 0.08   # Target=2: uniform over features
            ],
            "marginal": [0.33, 0.35, 0.32]
        }
    }
    
    results = {}
    target_entropy = it.entropy(target_dist)
    
    print(f"Target Entropy: {target_entropy:.3f} bits")
    print()
    
    for feature_name, data in features_vs_target.items():
        joint_dist = data["joint"]
        feature_dist = data["marginal"]
        
        # Calculate mutual information I(Feature; Target)
        mutual_info = it.mutual_information(joint_dist)
        
        # Calculate feature entropy
        feature_entropy = it.entropy(feature_dist)
        
        # Calculate conditional entropy H(Target|Feature)
        conditional_entropy = target_entropy - mutual_info
        
        # Information gain (same as mutual information for this case)
        info_gain = mutual_info
        
        # Normalized mutual information
        normalized_mi = mutual_info / min(target_entropy, feature_entropy) if min(target_entropy, feature_entropy) > 0 else 0
        
        results[feature_name] = {
            "mutual_information": mutual_info,
            "information_gain": info_gain,
            "normalized_mi": normalized_mi,
            "feature_entropy": feature_entropy,
            "conditional_entropy": conditional_entropy,
        }
        
        print(f"{feature_name}:")
        print(f"  Mutual Information: {mutual_info:.3f} bits")
        print(f"  Information Gain: {info_gain:.3f} bits")
        print(f"  Normalized MI: {normalized_mi:.3f}")
        print(f"  Feature Entropy: {feature_entropy:.3f} bits")
        print(f"  Conditional Entropy H(Target|Feature): {conditional_entropy:.3f} bits")
        print()
    
    # Rank features by mutual information
    ranked_features = sorted(
        results.items(),
        key=lambda x: x[1]["mutual_information"],
        reverse=True
    )
    
    print("🏆 Feature Ranking (by Mutual Information):")
    for i, (feature, metrics) in enumerate(ranked_features, 1):
        print(f"{i}. {feature}: {metrics['mutual_information']:.3f} bits")
    
    return results


def redundancy_analysis_example() -> Dict[str, float]:
    """
    Analyze feature redundancy using mutual information.
    
    Identifies redundant features that provide similar information,
    helping to reduce dimensionality while preserving information content.
    
    Returns:
        Dictionary with redundancy analysis results
    """
    print("\n🔄 Feature Redundancy Analysis")
    print("=" * 40)
    
    # Simulate a feature set with some redundant features
    features = {
        "Original_Feature": [0.4, 0.35, 0.25],
        "Highly_Redundant": [0.39, 0.36, 0.25],    # Almost identical
        "Moderately_Redundant": [0.5, 0.3, 0.2],   # Similar but different
        "Independent_Feature": [0.3, 0.3, 0.4],    # Different distribution
    }
    
    # Create joint distributions for feature pairs
    joint_distributions = {
        ("Original_Feature", "Highly_Redundant"): [
            0.38, 0.01, 0.01,  # Original=0: mostly Redundant=0
            0.01, 0.34, 0.00,  # Original=1: mostly Redundant=1
            0.00, 0.01, 0.24   # Original=2: mostly Redundant=2
        ],
        ("Original_Feature", "Moderately_Redundant"): [
            0.25, 0.10, 0.05,  # Some correlation
            0.15, 0.15, 0.05,
            0.10, 0.05, 0.10
        ],
        ("Original_Feature", "Independent_Feature"): [
            0.12, 0.12, 0.16,  # Independent
            0.105, 0.105, 0.14,
            0.075, 0.075, 0.10
        ],
    }
    
    results = {}
    
    for (feat1, feat2), joint_dist in joint_distributions.items():
        # Calculate mutual information
        mutual_info = it.mutual_information(joint_dist)
        
        # Calculate individual entropies
        entropy_1 = it.entropy(features[feat1])
        entropy_2 = it.entropy(features[feat2])
        
        # Normalized mutual information (redundancy measure)
        normalized_mi = mutual_info / min(entropy_1, entropy_2) if min(entropy_1, entropy_2) > 0 else 0
        
        # Redundancy score (0 = independent, 1 = perfectly redundant)
        redundancy_score = normalized_mi
        
        results[f"{feat1}_vs_{feat2}"] = {
            "mutual_information": mutual_info,
            "normalized_mi": normalized_mi,
            "redundancy_score": redundancy_score,
            "entropy_1": entropy_1,
            "entropy_2": entropy_2,
        }
        
        # Interpret redundancy level
        if redundancy_score > 0.8:
            redundancy_level = "HIGH - Consider removing one feature"
        elif redundancy_score > 0.5:
            redundancy_level = "MODERATE - Features share significant information"
        elif redundancy_score > 0.2:
            redundancy_level = "LOW - Features are somewhat related"
        else:
            redundancy_level = "MINIMAL - Features are largely independent"
        
        print(f"{feat1} vs {feat2}:")
        print(f"  Mutual Information: {mutual_info:.3f} bits")
        print(f"  Redundancy Score: {redundancy_score:.3f}")
        print(f"  Redundancy Level: {redundancy_level}")
        print()
    
    return results


def run_example() -> None:
    """
    Run all mutual information analysis examples.
    
    Demonstrates practical applications of mutual information in:
    - Feature dependency analysis
    - Feature selection and ranking
    - Redundancy detection
    """
    print("🔬 MetricaX Information Theory - Mutual Information Examples")
    print("=" * 65)
    
    # Run all examples
    dependency_results = feature_dependency_example()
    ranking_results = feature_selection_ranking_example()
    redundancy_results = redundancy_analysis_example()
    
    print("\n✅ All examples completed successfully!")
    print("\n💡 Key Takeaways:")
    print("• Higher mutual information = stronger feature dependency")
    print("• Use MI for feature selection: higher MI with target = better feature")
    print("• Detect redundant features: high MI between features = redundancy")
    print("• Normalized MI provides scale-independent comparison")


if __name__ == "__main__":
    run_example()
