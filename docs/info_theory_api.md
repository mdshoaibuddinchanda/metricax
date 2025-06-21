# Information Theory API Reference

Complete reference for MetricaX Information Theory module with mathematical formulas, parameters, and practical examples.

## Overview

The `metricax.info_theory` module provides 24 production-ready functions for information-theoretic analysis:

- **Entropy Measures** (6 functions): Shannon, cross-entropy, KL divergence, JS divergence, Renyi, Tsallis
- **Mutual Information** (7 functions): Mutual information, conditional entropy, information gain, dependencies
- **Distance Measures** (5 functions): Hellinger, total variation, Bhattacharyya, Wasserstein distances
- **Coding Theory** (3 functions): Optimal code length, Fano inequality, redundancy
- **Utilities** (3 functions): Distribution validation, safe logarithm, joint distributions

## Entropy and Variants

### `entropy(p)`

Compute Shannon entropy of a probability distribution.

**Formula**: `H(X) = -∑ p(x) log₂ p(x)`

**Parameters**:
- `p: List[float]` - Probability distribution (must sum to 1)

**Returns**: `float` - Entropy in bits

**Example**:
```python
import metricax.info_theory as it

# Maximum entropy (uniform distribution)
max_ent = it.entropy([0.25, 0.25, 0.25, 0.25])  # Returns: 2.0 bits

# Minimum entropy (deterministic)
min_ent = it.entropy([1.0, 0.0, 0.0, 0.0])      # Returns: 0.0 bits

# Typical case
typical = it.entropy([0.5, 0.3, 0.2])           # Returns: ~1.485 bits
```

**Use Cases**:
- Measuring uncertainty in data
- Feature selection (higher entropy = more informative)
- Model evaluation and comparison
- Data compression analysis

---

### `cross_entropy(p, q)`

Compute cross-entropy between two probability distributions.

**Formula**: `H(p, q) = -∑ p(x) log₂ q(x)`

**Parameters**:
- `p: List[float]` - True distribution
- `q: List[float]` - Predicted/model distribution

**Returns**: `float` - Cross-entropy in bits

**Example**:
```python
# Model evaluation
true_dist = [0.7, 0.2, 0.1]
model_pred = [0.6, 0.3, 0.1]

cross_ent = it.cross_entropy(true_dist, model_pred)  # Returns: ~1.13 bits

# Perfect model (identical distributions)
perfect = it.cross_entropy(true_dist, true_dist)    # Returns: entropy(true_dist)
```

**Use Cases**:
- Machine learning loss functions
- Model performance evaluation
- Information-theoretic model comparison
- Neural network training

---

### `kl_divergence(p, q)`

Compute Kullback-Leibler divergence between distributions.

**Formula**: `D(p || q) = ∑ p(x) log₂(p(x) / q(x))`

**Parameters**:
- `p: List[float]` - Reference distribution
- `q: List[float]` - Comparison distribution

**Returns**: `float` - KL divergence in bits (≥ 0)

**Example**:
```python
# Distribution comparison
p = [0.5, 0.3, 0.2]
q = [0.33, 0.33, 0.34]  # More uniform

kl_div = it.kl_divergence(p, q)  # Returns: ~0.097 bits

# Asymmetric property
reverse_kl = it.kl_divergence(q, p)  # Different value!

# Identical distributions
identical = it.kl_divergence(p, p)   # Returns: 0.0
```

**Use Cases**:
- Model comparison and selection
- Distribution fitting evaluation
- Variational inference
- Anomaly detection

---

### `js_divergence(p, q)`

Compute Jensen-Shannon divergence (symmetric version of KL).

**Formula**: `JS(p, q) = ½D(p || M) + ½D(q || M)` where `M = ½(p + q)`

**Parameters**:
- `p: List[float]` - First distribution
- `q: List[float]` - Second distribution

**Returns**: `float` - JS divergence in bits (0 ≤ JS ≤ 1)

**Example**:
```python
# Symmetric distance measure
p = [0.7, 0.2, 0.1]
q = [0.1, 0.2, 0.7]

js_div = it.js_divergence(p, q)      # Returns: ~0.58 bits
js_reverse = it.js_divergence(q, p)  # Same value (symmetric)

# Maximum divergence
max_div = it.js_divergence([1,0], [0,1])  # Returns: 1.0 bit
```

**Use Cases**:
- Clustering algorithms
- Phylogenetic analysis
- Document similarity
- Distribution comparison

---

### `renyi_entropy(p, alpha)`

Compute Rényi entropy of order α.

**Formula**: `H_α(X) = (1/(1-α)) log₂(∑ p(x)^α)` for α ≠ 1

**Parameters**:
- `p: List[float]` - Probability distribution
- `alpha: float` - Order parameter (α ≥ 0, α ≠ 1)

**Returns**: `float` - Rényi entropy in bits

**Example**:
```python
dist = [0.5, 0.3, 0.2]

# Different orders
h0 = it.renyi_entropy(dist, 0)    # Log of support size
h2 = it.renyi_entropy(dist, 2)    # Collision entropy
h_inf = it.renyi_entropy(dist, 100)  # Min-entropy (approx)

# Shannon entropy limit
h1_approx = it.renyi_entropy(dist, 1.001)  # ≈ Shannon entropy
```

**Use Cases**:
- Generalized entropy measures
- Cryptographic applications
- Diversity indices in ecology
- Fractal dimension analysis

---

### `tsallis_entropy(p, q)`

Compute Tsallis entropy (non-additive generalization).

**Formula**: `S_q(X) = (1/(q-1))(1 - ∑ p(x)^q)` for q ≠ 1

**Parameters**:
- `p: List[float]` - Probability distribution
- `q: float` - Entropic index (q > 0, q ≠ 1)

**Returns**: `float` - Tsallis entropy

**Example**:
```python
dist = [0.4, 0.3, 0.2, 0.1]

# Different q values
s0 = it.tsallis_entropy(dist, 0)    # Support size - 1
s2 = it.tsallis_entropy(dist, 2)    # 1 - ∑p²
s_half = it.tsallis_entropy(dist, 0.5)  # Super-additive case

# Shannon limit
s1_approx = it.tsallis_entropy(dist, 1.001)  # ≈ Shannon entropy
```

**Use Cases**:
- Non-extensive statistical mechanics
- Complex systems analysis
- Multifractal analysis
- Anomalous diffusion studies

## Mutual Information and Dependencies

### `mutual_information(joint_dist)`

Compute mutual information between two variables.

**Formula**: `I(X;Y) = ∑∑ p(x,y) log₂(p(x,y) / (p(x)p(y)))`

**Parameters**:
- `joint_dist: List[float]` - Joint probability distribution p(x,y)

**Returns**: `float` - Mutual information in bits (≥ 0)

**Example**:
```python
# Independent variables (MI = 0)
independent = [0.25, 0.25, 0.25, 0.25]  # 2x2 joint distribution
mi_indep = it.mutual_information(independent)  # Returns: 0.0

# Perfectly dependent variables
dependent = [0.5, 0.0, 0.0, 0.5]  # Perfect correlation
mi_dep = it.mutual_information(dependent)  # Returns: 1.0 bit

# Partial dependence
partial = [0.4, 0.1, 0.1, 0.4]
mi_partial = it.mutual_information(partial)  # Returns: ~0.278 bits
```

**Use Cases**:
- Feature selection in machine learning
- Variable dependency analysis
- Network analysis
- Causal inference

---

### `conditional_entropy(joint_dist)`

Compute conditional entropy H(Y|X).

**Formula**: `H(Y|X) = H(X,Y) - H(X)`

**Parameters**:
- `joint_dist: List[float]` - Joint probability distribution

**Returns**: `float` - Conditional entropy in bits

**Example**:
```python
# High conditional entropy (Y uncertain given X)
high_cond = [0.25, 0.25, 0.25, 0.25]
h_high = it.conditional_entropy(high_cond)  # Returns: 1.0 bit

# Low conditional entropy (Y predictable from X)
low_cond = [0.45, 0.05, 0.05, 0.45]
h_low = it.conditional_entropy(low_cond)   # Returns: ~0.286 bits
```

**Use Cases**:
- Predictability analysis
- Information gain calculation
- Decision tree construction
- Uncertainty quantification

---

### `information_gain(joint_dist)`

Compute information gain (mutual information).

**Formula**: `IG(Y|X) = H(Y) - H(Y|X) = I(X;Y)`

**Parameters**:
- `joint_dist: List[float]` - Joint probability distribution

**Returns**: `float` - Information gain in bits

**Example**:
```python
# Feature selection example
feature_target_joint = [0.3, 0.1, 0.2, 0.4]  # Feature vs target
info_gain = it.information_gain(feature_target_joint)

if info_gain > 0.1:
    print("Feature is informative for prediction")
else:
    print("Feature provides little information")
```

**Use Cases**:
- Feature ranking and selection
- Decision tree splitting criteria
- Attribute evaluation
- Dimensionality reduction

## Distance Measures

### `hellinger_distance(p, q)`

Compute Hellinger distance between distributions.

**Formula**: `H(p,q) = (1/√2) √(∑(√p(x) - √q(x))²)`

**Parameters**:
- `p: List[float]` - First distribution
- `q: List[float]` - Second distribution

**Returns**: `float` - Hellinger distance (0 ≤ H ≤ 1)

**Example**:
```python
# Distribution comparison
p = [0.6, 0.3, 0.1]
q = [0.4, 0.4, 0.2]

hell_dist = it.hellinger_distance(p, q)  # Returns: ~0.141

# Identical distributions
identical = it.hellinger_distance(p, p)  # Returns: 0.0

# Maximum distance
max_dist = it.hellinger_distance([1,0], [0,1])  # Returns: 1.0
```

**Use Cases**:
- Distribution comparison
- Clustering algorithms
- Hypothesis testing
- Model validation

---

### `total_variation_distance(p, q)`

Compute total variation distance.

**Formula**: `TV(p,q) = ½ ∑|p(x) - q(x)|`

**Parameters**:
- `p: List[float]` - First distribution
- `q: List[float]` - Second distribution

**Returns**: `float` - Total variation distance (0 ≤ TV ≤ 1)

**Example**:
```python
# Close distributions
close_p = [0.5, 0.3, 0.2]
close_q = [0.45, 0.35, 0.2]
tv_close = it.total_variation_distance(close_p, close_q)  # Returns: 0.05

# Distant distributions
far_p = [0.8, 0.1, 0.1]
far_q = [0.1, 0.8, 0.1]
tv_far = it.total_variation_distance(far_p, far_q)  # Returns: 0.7
```

**Use Cases**:
- Statistical hypothesis testing
- Convergence analysis
- Approximation quality assessment
- Monte Carlo validation

## Coding Theory

### `optimal_code_length(p)`

Compute optimal average code length (Shannon's theorem).

**Formula**: `L* = H(X) = -∑ p(x) log₂ p(x)`

**Parameters**:
- `p: List[float]` - Source probability distribution

**Returns**: `float` - Optimal code length in bits per symbol

**Example**:
```python
# Uniform source (worst case for compression)
uniform = [0.25, 0.25, 0.25, 0.25]
uniform_length = it.optimal_code_length(uniform)  # Returns: 2.0 bits

# Skewed source (good for compression)
skewed = [0.5, 0.25, 0.125, 0.125]
skewed_length = it.optimal_code_length(skewed)   # Returns: 1.75 bits

# Compression ratio
compression_ratio = 2.0 / skewed_length  # 1.14x compression
```

**Use Cases**:
- Data compression analysis
- Communication system design
- Storage optimization
- Bandwidth calculation

---

### `fano_inequality(conditional_entropy, alphabet_size)`

Compute Fano bound on error probability.

**Formula**: `P_e ≥ (H(X|Y) - 1) / log₂(|X| - 1)`

**Parameters**:
- `conditional_entropy: float` - H(X|Y) in bits
- `alphabet_size: int` - Size of alphabet |X|

**Returns**: `float` - Lower bound on error probability

**Example**:
```python
# Communication channel analysis
h_xy = 1.5  # Conditional entropy
alphabet = 8  # 8 possible symbols

error_bound = it.fano_inequality(h_xy, alphabet)
print(f"Minimum error rate: {error_bound:.1%}")

# Perfect channel
perfect_bound = it.fano_inequality(0.0, alphabet)  # Returns: 0.0
```

**Use Cases**:
- Communication system limits
- Error correction bounds
- Channel capacity analysis
- Information-theoretic security

## Complete Example: Feature Selection Pipeline

```python
import metricax.info_theory as it

def select_features(features_data, target_data, top_k=5):
    """
    Select top-k features using information theory measures.
    
    Args:
        features_data: Dict of feature_name -> distribution
        target_data: Target variable distribution
        top_k: Number of features to select
    
    Returns:
        List of (feature_name, score) tuples
    """
    feature_scores = []
    target_entropy = it.entropy(target_data)
    
    for feature_name, feature_dist in features_data.items():
        # Calculate mutual information (requires joint distribution)
        # This is simplified - in practice, you'd compute from data
        
        # Simulate joint distribution based on correlation
        joint_dist = simulate_joint_distribution(feature_dist, target_data)
        
        # Calculate information measures
        mutual_info = it.mutual_information(joint_dist)
        info_gain = it.information_gain(joint_dist)
        
        # Feature entropy
        feature_entropy = it.entropy(feature_dist)
        
        # Normalized mutual information
        normalized_mi = mutual_info / min(target_entropy, feature_entropy)
        
        feature_scores.append((feature_name, {
            'mutual_information': mutual_info,
            'information_gain': info_gain,
            'normalized_mi': normalized_mi,
            'feature_entropy': feature_entropy
        }))
    
    # Sort by mutual information
    feature_scores.sort(key=lambda x: x[1]['mutual_information'], reverse=True)
    
    return feature_scores[:top_k]

# Example usage
features = {
    'age': [0.3, 0.4, 0.3],
    'income': [0.25, 0.5, 0.25],
    'education': [0.4, 0.3, 0.3],
    'location': [0.33, 0.33, 0.34]
}

target = [0.7, 0.3]  # Binary classification

top_features = select_features(features, target, top_k=3)
print("Top features for prediction:")
for name, scores in top_features:
    print(f"{name}: MI={scores['mutual_information']:.3f}")
```
