"""
Coding Theory Examples

Demonstrates practical applications of coding theory for:
- Optimal code length calculation
- Data compression analysis
- Communication channel capacity
- Error bounds and Fano's inequality

@formula: L* = H(X) (Shannon's source coding theorem)
@source: Cover & Thomas, Elements of Information Theory
"""

import random
from typing import List, Tuple, Dict
import metricax.info_theory as it


def optimal_coding_example() -> Dict[str, float]:
    """
    Demonstrate optimal coding length calculation for different distributions.
    
    Shows how entropy determines the theoretical minimum bits needed
    to encode messages from different probability distributions.
    
    Returns:
        Dictionary with coding analysis results
    """
    print("📡 Optimal Coding Length Analysis")
    print("=" * 40)
    
    # Different message distributions
    distributions = {
        "Uniform_4_Symbols": [0.25, 0.25, 0.25, 0.25],      # Maximum entropy
        "Skewed_Distribution": [0.5, 0.3, 0.15, 0.05],       # Common in natural language
        "Highly_Skewed": [0.8, 0.1, 0.06, 0.04],            # Very uneven
        "Binary_Balanced": [0.5, 0.5],                       # Coin flip
        "Binary_Biased": [0.9, 0.1],                         # Biased coin
        "English_Letters": [0.12, 0.09, 0.08, 0.075, 0.07, 0.065, 0.06, 0.055],  # Simplified
    }
    
    results = {}
    
    for dist_name, distribution in distributions.items():
        # Calculate entropy (theoretical minimum)
        entropy = it.entropy(distribution)
        
        # Calculate optimal code length
        optimal_length = it.optimal_code_length(distribution)
        
        # Fixed-length coding (naive approach)
        num_symbols = len(distribution)
        fixed_length = math.ceil(math.log2(num_symbols)) if num_symbols > 1 else 1
        
        # Compression ratio
        compression_ratio = fixed_length / optimal_length if optimal_length > 0 else 1
        
        # Efficiency (how close to theoretical minimum)
        efficiency = entropy / optimal_length if optimal_length > 0 else 1
        
        results[dist_name] = {
            "entropy": entropy,
            "optimal_length": optimal_length,
            "fixed_length": fixed_length,
            "compression_ratio": compression_ratio,
            "efficiency": efficiency,
            "num_symbols": num_symbols,
        }
        
        print(f"{dist_name}:")
        print(f"  Distribution: {[f'{p:.3f}' for p in distribution]}")
        print(f"  Entropy: {entropy:.3f} bits/symbol")
        print(f"  Optimal Code Length: {optimal_length:.3f} bits/symbol")
        print(f"  Fixed-Length Code: {fixed_length} bits/symbol")
        print(f"  Compression Ratio: {compression_ratio:.2f}x")
        print(f"  Coding Efficiency: {efficiency:.1%}")
        print()
    
    return results


def compression_analysis_example() -> Dict[str, float]:
    """
    Analyze compression potential for different types of data.
    
    Demonstrates how entropy predicts compression limits for various
    data types commonly encountered in practice.
    
    Returns:
        Dictionary with compression analysis results
    """
    print("🗜️ Data Compression Analysis")
    print("=" * 35)
    
    # Simulate different data types with their characteristic distributions
    data_types = {
        "Random_Data": [1/256] * 256,  # Uniform distribution (worst case for compression)
        "Natural_Language": [
            0.12, 0.09, 0.08, 0.075, 0.07, 0.065, 0.06, 0.055,  # Common letters
            0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015,  # Less common
            0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003,  # Rare
            0.002, 0.001  # Very rare
        ],
        "DNA_Sequence": [0.25, 0.25, 0.25, 0.25],  # A, T, G, C (approximately uniform)
        "Biased_DNA": [0.4, 0.3, 0.2, 0.1],        # GC-rich regions
        "Sensor_Data": [
            0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.015, 0.005  # Typical sensor readings
        ],
        "Network_Traffic": [
            0.4, 0.2, 0.15, 0.1, 0.08, 0.04, 0.02, 0.01  # Packet size distribution
        ],
    }
    
    results = {}
    
    for data_type, distribution in data_types.items():
        # Calculate entropy
        entropy = it.entropy(distribution)
        
        # Maximum possible entropy for this alphabet size
        alphabet_size = len(distribution)
        max_entropy = math.log2(alphabet_size)
        
        # Compression potential
        compression_potential = 1 - (entropy / max_entropy)
        
        # Theoretical compression ratio
        uncompressed_bits = max_entropy  # Fixed-length encoding
        compressed_bits = entropy        # Optimal encoding
        theoretical_ratio = uncompressed_bits / compressed_bits if compressed_bits > 0 else 1
        
        # Practical compression estimate (accounting for overhead)
        practical_ratio = theoretical_ratio * 0.85  # 85% of theoretical due to overhead
        
        results[data_type] = {
            "entropy": entropy,
            "max_entropy": max_entropy,
            "compression_potential": compression_potential,
            "theoretical_ratio": theoretical_ratio,
            "practical_ratio": practical_ratio,
            "alphabet_size": alphabet_size,
        }
        
        print(f"{data_type}:")
        print(f"  Alphabet Size: {alphabet_size} symbols")
        print(f"  Entropy: {entropy:.3f} bits/symbol")
        print(f"  Maximum Entropy: {max_entropy:.3f} bits/symbol")
        print(f"  Compression Potential: {compression_potential:.1%}")
        print(f"  Theoretical Compression: {theoretical_ratio:.2f}x")
        print(f"  Practical Compression: {practical_ratio:.2f}x")
        print()
    
    return results


def channel_capacity_example() -> Dict[str, float]:
    """
    Demonstrate channel capacity calculation using information theory.
    
    Shows how to calculate the maximum information transmission rate
    for different types of communication channels.
    
    Returns:
        Dictionary with channel capacity analysis
    """
    print("📶 Communication Channel Capacity")
    print("=" * 40)
    
    # Different channel types with their error characteristics
    channels = {
        "Perfect_Channel": {
            "error_rate": 0.0,
            "description": "No errors (theoretical ideal)"
        },
        "Low_Noise_Channel": {
            "error_rate": 0.01,
            "description": "1% bit error rate (high-quality fiber)"
        },
        "Moderate_Noise_Channel": {
            "error_rate": 0.05,
            "description": "5% bit error rate (wireless in good conditions)"
        },
        "High_Noise_Channel": {
            "error_rate": 0.1,
            "description": "10% bit error rate (poor wireless)"
        },
        "Very_Noisy_Channel": {
            "error_rate": 0.2,
            "description": "20% bit error rate (very poor conditions)"
        },
    }
    
    results = {}
    
    for channel_name, channel_data in channels.items():
        error_rate = channel_data["error_rate"]
        description = channel_data["description"]
        
        # Binary symmetric channel capacity
        if error_rate == 0:
            capacity = 1.0  # Perfect channel
        elif error_rate == 0.5:
            capacity = 0.0  # Completely noisy channel
        else:
            # C = 1 - H(p) where p is error probability
            error_entropy = it.entropy([error_rate, 1 - error_rate])
            capacity = 1.0 - error_entropy
        
        # Maximum reliable transmission rate
        max_rate_bps = capacity  # bits per channel use
        
        # Redundancy needed for error correction
        redundancy = 1 - capacity if capacity > 0 else 1
        
        # Effective throughput (accounting for error correction overhead)
        effective_throughput = capacity * 0.9  # 90% efficiency due to practical constraints
        
        results[channel_name] = {
            "error_rate": error_rate,
            "capacity": capacity,
            "max_rate_bps": max_rate_bps,
            "redundancy": redundancy,
            "effective_throughput": effective_throughput,
            "description": description,
        }
        
        print(f"{channel_name}:")
        print(f"  Description: {description}")
        print(f"  Error Rate: {error_rate:.1%}")
        print(f"  Channel Capacity: {capacity:.3f} bits/transmission")
        print(f"  Required Redundancy: {redundancy:.1%}")
        print(f"  Effective Throughput: {effective_throughput:.3f} bits/transmission")
        print()
    
    return results


def fano_bound_example() -> Dict[str, float]:
    """
    Demonstrate Fano's inequality for error bound analysis.
    
    Shows how Fano's inequality provides fundamental limits on
    the probability of error in communication and estimation.
    
    Returns:
        Dictionary with Fano bound analysis
    """
    print("⚠️ Fano's Inequality - Error Bounds")
    print("=" * 40)
    
    # Different scenarios with varying uncertainty
    scenarios = {
        "High_Certainty": {
            "num_outcomes": 4,
            "conditional_entropy": 0.5,  # Low uncertainty
            "description": "Clear signal, low noise"
        },
        "Moderate_Certainty": {
            "num_outcomes": 8,
            "conditional_entropy": 1.5,  # Moderate uncertainty
            "description": "Some noise, moderate confusion"
        },
        "Low_Certainty": {
            "num_outcomes": 16,
            "conditional_entropy": 3.0,  # High uncertainty
            "description": "High noise, significant confusion"
        },
        "Very_Low_Certainty": {
            "num_outcomes": 32,
            "conditional_entropy": 4.5,  # Very high uncertainty
            "description": "Very noisy channel"
        },
    }
    
    results = {}
    
    for scenario_name, scenario_data in scenarios.items():
        num_outcomes = scenario_data["num_outcomes"]
        conditional_entropy = scenario_data["conditional_entropy"]
        description = scenario_data["description"]
        
        # Apply Fano's inequality: H(X|Y) ≤ H(Pe) + Pe * log(|X| - 1)
        # Solve for Pe (probability of error)
        
        # Fano bound calculation
        fano_bound = it.fano_inequality(conditional_entropy, num_outcomes)
        
        # Theoretical minimum error rate
        min_error_rate = fano_bound
        
        # Maximum achievable accuracy
        max_accuracy = 1 - min_error_rate
        
        results[scenario_name] = {
            "num_outcomes": num_outcomes,
            "conditional_entropy": conditional_entropy,
            "fano_bound": fano_bound,
            "min_error_rate": min_error_rate,
            "max_accuracy": max_accuracy,
            "description": description,
        }
        
        print(f"{scenario_name}:")
        print(f"  Description: {description}")
        print(f"  Number of Outcomes: {num_outcomes}")
        print(f"  Conditional Entropy: {conditional_entropy:.3f} bits")
        print(f"  Fano Bound (Min Error Rate): {fano_bound:.3f}")
        print(f"  Maximum Achievable Accuracy: {max_accuracy:.1%}")
        print()
    
    return results


def run_example() -> None:
    """
    Run all coding theory examples.
    
    Demonstrates practical applications of coding theory in:
    - Optimal code length calculation
    - Data compression analysis
    - Channel capacity estimation
    - Error bound analysis with Fano's inequality
    """
    print("🔬 MetricaX Information Theory - Coding Theory Examples")
    print("=" * 60)
    
    # Import math for calculations
    import math
    
    # Run all examples
    coding_results = optimal_coding_example()
    compression_results = compression_analysis_example()
    capacity_results = channel_capacity_example()
    fano_results = fano_bound_example()
    
    print("\n✅ All examples completed successfully!")
    print("\n💡 Key Takeaways:")
    print("• Entropy determines optimal code length (Shannon's theorem)")
    print("• Lower entropy = better compression potential")
    print("• Channel capacity limits reliable transmission rate")
    print("• Fano's inequality provides fundamental error bounds")


if __name__ == "__main__":
    run_example()
