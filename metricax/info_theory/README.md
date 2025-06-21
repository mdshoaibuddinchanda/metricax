# MetricaX Information Theory Module 📡

**Comprehensive information-theoretic analysis toolkit for machine learning and data science**

This module provides production-ready implementations of fundamental information theory concepts, from basic entropy measures to advanced distribution comparison metrics.

## 🚀 **Overview**

The Information Theory module covers the mathematical foundations of information, uncertainty, and communication with applications in:
- **Machine Learning** - Loss functions, model comparison, feature selection
- **Data Compression** - Optimal coding, redundancy analysis
- **Statistical Analysis** - Distribution comparison, dependence measurement
- **Communication Theory** - Channel capacity, error bounds

## 📊 **Function Categories**

### **Entropy and Variants (6 functions)**
Fundamental measures of uncertainty and information content:

| Function | Mathematical Form | Applications |
|----------|-------------------|--------------|
| `entropy(p)` | H(X) = -Σp(x)log p(x) | Baseline uncertainty measure |
| `cross_entropy(p, q)` | H(p,q) = -Σp(x)log q(x) | ML loss functions, model evaluation |
| `kl_divergence(p, q)` | D_KL(p‖q) = Σp(x)log(p(x)/q(x)) | Model comparison, variational inference |
| `js_divergence(p, q)` | Symmetric KL divergence | Balanced distribution comparison |
| `renyi_entropy(p, α)` | H_α(X) = (1/(1-α))log(Σp^α) | Generalized entropy measures |
| `tsallis_entropy(p, q)` | S_q = (1-Σp^q)/(q-1) | Non-extensive statistical mechanics |

### **Mutual Information & Dependence (7 functions)**
Measures of statistical dependence and shared information:

| Function | Purpose | Use Cases |
|----------|---------|-----------|
| `mutual_information(p_xy, p_x, p_y)` | I(X;Y) quantifies dependence | Feature selection, clustering |
| `conditional_entropy(p_xy, p_y)` | H(X\|Y) remaining uncertainty | Predictive modeling |
| `information_gain(p_prior, p_posterior)` | Entropy reduction | Decision trees, active learning |
| `symmetric_uncertainty(p_xy, p_x, p_y)` | Normalized MI ∈ [0,1] | Feature ranking |
| `variation_of_information(p_xy, p_x, p_y)` | VI metric distance | Clustering validation |
| `total_correlation(p_xyz, p_x, p_y, p_z)` | Multi-variate dependence | Complex system analysis |
| `multi_information(p_joint, marginals)` | Alternative total correlation | High-dimensional analysis |

### **Coding Theory (3 functions)**
Optimal coding and information-theoretic bounds:

| Function | Theoretical Foundation | Applications |
|----------|----------------------|--------------|
| `optimal_code_length(p)` | Shannon's source coding | Data compression, Huffman coding |
| `fano_inequality(h_xy, error, alphabet)` | Error probability bounds | Communication limits |
| `redundancy(p, code_lengths)` | Coding efficiency | Compression performance |

### **Distance Measures (4 functions)**
Metrics for comparing probability distributions:

| Function | Properties | Best For |
|----------|------------|----------|
| `hellinger_distance(p, q)` | Bounded [0,1], symmetric, metric | General distribution comparison |
| `total_variation_distance(p, q)` | Max probability difference | Discrete distributions |
| `bhattacharyya_distance(p, q)` | Overlap-based measure | Classification, clustering |
| `wasserstein_distance_1d(p, q)` | Earth mover's distance | Ordered data, optimal transport |

### **Utilities (4 functions)**
Mathematical support with numerical stability:

| Function | Purpose | Features |
|----------|---------|----------|
| `validate_distribution(p)` | Input validation | Comprehensive error checking |
| `normalize_distribution(p)` | Probability normalization | Automatic sum-to-one |
| `joint_distribution(p_x, p_y_x)` | Joint from marginal/conditional | Distribution construction |
| `safe_log(x, base, epsilon)` | Numerically stable logarithm | Handles edge cases |

## 🎯 **Real-World Applications**

### **Feature Selection with Mutual Information**
```python
import metricax.info_theory as it

# Calculate MI between features and target
def select_features(features, target, threshold=0.1):
    selected = []
    for i, feature in enumerate(features):
        # Compute joint and marginal distributions
        joint_dist = compute_joint_distribution(feature, target)
        marginal_feature = compute_marginal(joint_dist, axis=1)
        marginal_target = compute_marginal(joint_dist, axis=0)
        
        # Calculate mutual information
        mi = it.mutual_information(joint_dist, marginal_feature, marginal_target)
        
        if mi > threshold:
            selected.append(i)
    
    return selected
```

### **Model Comparison with KL Divergence**
```python
# Compare model predictions to true distribution
true_dist = [0.7, 0.2, 0.1]
model_a_pred = [0.65, 0.25, 0.1]
model_b_pred = [0.5, 0.3, 0.2]

kl_a = it.kl_divergence(true_dist, model_a_pred)
kl_b = it.kl_divergence(true_dist, model_b_pred)

print(f"Model A KL: {kl_a:.3f}, Model B KL: {kl_b:.3f}")
# Lower KL divergence indicates better model
```

### **Cross-Entropy Loss for Classification**
```python
# Multi-class classification loss
def cross_entropy_loss(y_true, y_pred):
    """Compute cross-entropy loss for classification."""
    total_loss = 0
    for true_dist, pred_dist in zip(y_true, y_pred):
        # Normalize predictions to probabilities
        pred_normalized = it.normalize_distribution(pred_dist)
        loss = it.cross_entropy(true_dist, pred_normalized)
        total_loss += loss
    
    return total_loss / len(y_true)
```

### **Data Compression Analysis**
```python
# Analyze compression efficiency
symbol_probs = [0.5, 0.25, 0.125, 0.125]  # Symbol probabilities
actual_lengths = [1, 2, 3, 3]  # Actual code lengths

# Optimal (theoretical) code lengths
optimal_lengths = it.optimal_code_length(symbol_probs)

# Compression efficiency
redundancy = it.redundancy(symbol_probs, actual_lengths)
efficiency = 1 - redundancy / it.entropy(symbol_probs)

print(f"Compression efficiency: {efficiency:.2%}")
```

## 📈 **Mathematical Foundations**

### **Shannon Entropy**
The fundamental measure of uncertainty:
```
H(X) = -Σ p(x) log₂ p(x)
```
- Measured in bits (base 2), nats (base e), or dits (base 10)
- Maximum for uniform distributions
- Zero for deterministic outcomes

### **Mutual Information**
Quantifies shared information between variables:
```
I(X;Y) = Σ p(x,y) log(p(x,y)/(p(x)p(y)))
      = H(X) - H(X|Y)
      = H(Y) - H(Y|X)
```

### **KL Divergence**
Measures "distance" from distribution q to p:
```
D_KL(p||q) = Σ p(x) log(p(x)/q(x))
```
- Non-negative, zero iff p = q
- Asymmetric: D_KL(p||q) ≠ D_KL(q||p)
- Related to cross-entropy: H(p,q) = H(p) + D_KL(p||q)

## 🧪 **Advanced Features**

### **Numerical Stability**
- Safe logarithm computation with epsilon handling
- Overflow protection for extreme probability values
- Robust handling of zero probabilities

### **Multiple Logarithm Bases**
- Base 2 (bits) - Default for information theory
- Base e (nats) - Natural logarithm for mathematical analysis
- Base 10 (dits) - Decimal information units
- Custom bases for specialized applications

### **Distribution Validation**
- Automatic probability distribution validation
- Comprehensive error messages
- Tolerance settings for numerical precision

## 🔬 **Performance Characteristics**

### **Computational Complexity**
- Most functions: O(n) where n is distribution size
- Joint distribution functions: O(n×m) for n×m distributions
- Memory efficient: minimal temporary storage

### **Numerical Precision**
- IEEE 754 double precision
- Epsilon-based stability (default: 1e-15)
- Graceful handling of edge cases

## 📚 **References**

- **Cover, T.M. & Thomas, J.A.** (2006). *Elements of Information Theory*. 2nd Edition.
- **MacKay, D.J.C.** (2003). *Information Theory, Inference and Learning Algorithms*.
- **Shannon, C.E.** (1948). "A Mathematical Theory of Communication". *Bell System Technical Journal*.

## 🤝 **Contributing**

See the main [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines on adding new information theory functions or improving existing implementations.

---

**Part of MetricaX - Professional Mathematical Toolkit for Python**
